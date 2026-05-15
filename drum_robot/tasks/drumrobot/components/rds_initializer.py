
"""
목표 악보(RDS: robotic drum score) 초기화 클래스 

path: source/extensions/drum_robot/drum_robot/tasks/drumrobot/components/rds_initializer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from mido import MidiFile
import torch
from pathlib import Path
import random
import time

GENERAL_MIDI_PERCUSSION_KEY_MAP = {
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    37: "Side Stick",
    38: "Acoustic Snare",
    39: "Hand Clap",
    40: "Electric Snare",
    41: "Low Floor Tom",
    42: "Closed Hi Hat",
    43: "High Floor Tom",
    44: "Pedal Hi-Hat",
    45: "Low Tom",
    46: "Open Hi-Hat",
    47: "Low-Mid Tom",
    48: "Hi Mid Tom",
    49: "Crash Cymbal 1",
    50: "High Tom",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    54: "Tambourine",
    55: "Splash Cymbal",
    56: "Cowbell",
    57: "Crash Cymbal 2",
    59: "Ride Cymbal 2",
    # 60: "Hi Bongo",
    # 61: "Low Bongo",
    # 62: "Mute Hi Conga",
    # 63: "Open Hi Conga",
    # 64: "Low Conga",
    # 65: "High Timbale",
    # 66: "Low Timbale",
    # 67: "High Agogo",
    # 68: "Low Agogo",
    # 69: "Cabasa",
    # 70: "Maracas",
    # 71: "Short Whistle",
    # 72: "Long Whistle",
    # 73: "Short Guiro",
    # 74: "Long Guiro",
    # 75: "Claves",
    # 76: "Hi Wood Block",
    # 77: "Low Wood Block",
    # 78: "Mute Cuica",
    # 79: "Open Cuica",
    # 80: "Mute Triangle",
    # 81: "Open Triangle",
    22: "Tom(22)",
    26: "Tom(26)",
    58: "58",
}

MIDI_FOLDER_PATH = "/home/shy/RL_workspace/IsaacLab/source/extensions/drum_robot/drum_robot/MIDIs"

@dataclass
class RdsInitializerCfg:

    # 에피소드 시간
    episode_length_s: float = 5.0

    # 스텝 시간
    dt: float = 1/60

    # 드럼 악기의 개수
    num_drum: int = 8

    slow_factor: float = 2.0   # 0.5배속
    start_offset_steps: int = 20
    hit_window_step: int = 5

    instrument_to_idx = {
        "Acoustic Snare": 0,
        "Electric Snare": 0,
        "Side Stick": 0,    # 임시로 스네어 타격

        "Low Floor Tom": 1,
        "High Floor Tom": 1,
        
        "Low Tom": 2,
        "Low-Mid Tom": 2,

        "Hi Mid Tom": 3,
        "High Tom": 3,

        "Closed Hi Hat": 4,
        "Open Hi-Hat": 4,
        
        "Ride Cymbal 1": 5,
        "Ride Cymbal 2": 5,
        "Ride Bell": 5,     # 임시로 라이드 타격

        "Crash Cymbal 1": 7,    # 왼쪽 크러시
        "Crash Cymbal 2": 7,
        "Chinese Cymbal": 6,    # 임시로 오른쪽 크러시 타격
        "Splash Cymbal": 6,

        # 알 수 없는 거
        "Tom(22)": 3,
        "Tom(26)": 3,   # 확인 필요
    }

    instrument_pedal = {
        # 페달로 타격하는 경우 예외 처리
        "Acoustic Bass Drum",
        "Bass Drum 1",
        "Pedal Hi-Hat",
        "58",   # 뭔지 모르겠음
    }

class RdsInitializer:

    def __init__(self, num_envs: int, device: torch.device | str, cfg: RdsInitializerCfg):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cfg = cfg
        self.episode_length = int(self.cfg.episode_length_s / self.cfg.dt)

        random.seed(time.time_ns())
        self.rng = random.Random()

        self.midi_files = self._glob_midi_files()
        self.rds = self._build_rds_dataset(self.midi_files)
        self.score = self._compute_score(self.rds)                     # (N,)
    
    def _glob_midi_files(self):
        folder = Path(MIDI_FOLDER_PATH)
        midi_files = list(folder.glob("*.mid"))

        if not midi_files:
            raise ValueError("MIDI 파일이 없습니다!")

        return midi_files
    
    def _build_rds_dataset(self, midi_files):
        rds_list = []

        for file_path in midi_files:
            self._read_midi_file(file_path)

            end_event = False
            prev_t = 0.0

            while not end_event:
                rds, t, end_event = self._generate_rds(prev_t)
                prev_t = t

                if torch.any(rds > 0):
                    rds_list.append(rds)

        if len(rds_list) == 0:
            raise RuntimeError("No RDS segments were generated from MIDI files.")

        rds_tensors = torch.stack(rds_list, dim=0).to(self.device)   # (N, T, M)
        
        return rds_tensors

    def _read_midi_file(self, file_path):
        mid = MidiFile(file_path)
        ticks_per_beat = mid.ticks_per_beat

        default_tempo = 500000  # 120 BPM
        tempo = default_tempo
        bpm = 60_000_000 / tempo

        events = []

        # 먼저 전체 트랙에서 템포 찾기
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    bpm = 60_000_000 / tempo
                    break
            if tempo != default_tempo:
                break

        # print(f"Detected BPM: {bpm:.2f}")
        self.bpm = bpm

        # 이벤트 파싱
        for i, track in enumerate(mid.tracks):  # 대부분 단일 트랙인거 같으니 드럼인지 확인하고 하나만 읽자
            current_time = 0
            tempo = default_tempo  # 트랙별 초기화

            for msg in track:
                current_time += msg.time

                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    # print(f"Track: {i}, bpm: {60_000_000 / tempo}")

                if msg.type == 'note_on' and msg.velocity > 0 and msg.channel == 9:
                    note = msg.note
                    instrument = GENERAL_MIDI_PERCUSSION_KEY_MAP.get(note, f"Unknown({note})")

                    time_sec = (current_time * tempo) / (ticks_per_beat * 1_000_000)

                    events.append((time_sec, instrument))

        events.sort()   # 멀티 트랙인 경우 시간 순으로 정렬

        # for t, inst in events:
        #     print(f"{t:.3f}s : {inst}")
        self.events = events
    
    def _generate_rds(self, prev_t):
        T = self.episode_length
        M = self.cfg.num_drum

        bpm = self.bpm
        events = self.events

        seconds_per_beat = 60.0 / bpm
        measure_duration = 4 * seconds_per_beat  # 4/4

        robotic_drum_score = torch.zeros((T, M), device=self.device, dtype=torch.int64)

        first = True
        end = True
        next_t = 0.0

        for t, inst in events:
            if t < prev_t:
                continue

            if first:
                first_t = t
                first = False

            if t > first_t + measure_duration:
                end = False
                next_t = t
                break

            # instrument → index
            if inst not in self.cfg.instrument_to_idx:
                if inst not in self.cfg.instrument_pedal:
                    print(f"Warning: Unknown instrument {inst}, skipping.")
                continue
            inst_idx = self.cfg.instrument_to_idx[inst]

            # 시간 → index
            time_idx = int(round(self.cfg.slow_factor * (t - first_t) / self.cfg.dt)) + self.cfg.start_offset_steps
            safe_T = T - self.cfg.hit_window_step   # episode 마지막 W step은 판정하지 않음으로 타격으로 채우지 않음
            if time_idx >= safe_T:
                continue  # episode 길이 초과 이벤트는 무시

            # 모든 env에 이벤트 적용
            # print(f"robotic_drum_score: {time_idx}/{T}, {inst_idx+1}")
            robotic_drum_score[time_idx, inst_idx] = 1

        return robotic_drum_score, next_t, end

    def _compute_score(self, rds_tensors: torch.Tensor) -> torch.Tensor:
        # rds_tensors: (N, T, M), 0/1
        N, T, M = rds_tensors.shape

        # 1. 사용된 드럼 개수
        num_drum_hit = rds_tensors.sum(dim=1)   # (N, M)
        drum_hit_mask = num_drum_hit > 0.5
        num_target = drum_hit_mask.float().sum(dim=1)   # (N,)

        # 2. 드럼 변경 횟수
        switch_count = torch.zeros((N,), device=self.device)

        prev_inst = torch.zeros((N, M), device=self.device)
        for i in range(T):
            curr_inst = rds_tensors[:, i, :]  # (N, M)
            curr_hit = curr_inst.sum(dim=1) > 0.5  # (N,)

            # 이전과 현재가 다르면 switch
            diff = (curr_inst != prev_inst).any(dim=1)  # (N,)
            switch = curr_hit & diff
            switch_count += switch.float()

            # 현재 타격이 있을 때만 prev 갱신
            prev_inst = torch.where(curr_hit.unsqueeze(1), curr_inst, prev_inst)

        # 3. 스네어 비율
        num_total_hit = num_drum_hit.sum(dim=1).clamp(min=1e-6)
        snare_rate = num_drum_hit[:, 0] / num_total_hit

        # 4. 크러시
        crash_count = num_drum_hit[:, 6] + num_drum_hit[:, 7]
        crash_exist = (crash_count > 0).float()

        # 5. 최종 score
        # w1, w2, w3, w4 = 1.0, 0.5, 2.0, 1.0
        w1, w2, w3, w4 = 1.0, 1.0, 1.0, 0.0
        score = w1 * num_target + w2 * switch_count + w3 * (1 - snare_rate) + w4 * crash_exist

        # print(f"num_target: {num_target.sum()/N}")
        # print(f"switch_count: {switch_count.sum()/N}")
        # print(f"snare_rate: {snare_rate.sum()/N}")
        # print(f"crash_rate: {crash_exist.sum()/N}")

        return score

    def _reset_target_midi(self, env_ids, selection_strength=0.5):
        weights = self.score.clone()

        # selection_strength 랜덤성 조절 (0 <= selection_strength < 1)
        # 값이 클수록 스코어 기준으로 선별, 값이 0이면 완전 랜덤
        weights = torch.pow(weights, selection_strength)

        probs = weights / (weights.sum() + 1e-6)

        # probs를 기반으로 인덱스를 랜덤 샘플링
        idx = torch.multinomial(
            probs,
            num_samples=len(env_ids),
            replacement=True    # 같은 인덱스가 여러 번 뽑힐 수 있음
        )

        return self.rds[idx]

    def _reset_target_rand(self, env_ids):
        N = len(env_ids)
        T = self.episode_length
        M = self.cfg.num_drum

        rds_rand = torch.zeros((N, T, M), device=self.device, dtype=torch.int64)

        s = self.cfg.start_offset_steps
        e = T - self.cfg.hit_window_step

        k = 10
        for i in range(k):
            si = (int)(s + (2 * i) * (e - s) / (2 * k - 1))
            ei = (int)(s + (2 * i + 1) * (e - s) / (2 * k - 1))
            time_rand = torch.randint(si, ei, (N,), device=self.device)
            inst_rand = torch.randint(0, M, (N,), device=self.device)

            rds_rand[torch.arange(N), time_rand, inst_rand] = 1

        return rds_rand
    
    def reset_target(self, env_ids, score_ratio=0.5, selection_strength=0.5):
        N = len(env_ids)

        num_score = int(N * score_ratio)

        perm = torch.randperm(N, device=self.device)    # 0~N 사이의 정수를 무작위로 섞어서 텐서 만들기
        score_env_ids = env_ids[perm[:num_score]]
        rand_env_ids = env_ids[perm[num_score:]]

        outputs = []

        # -------------------
        # score-based
        # -------------------
        if num_score > 0:
            out_score = self._reset_target_midi(
                score_env_ids,
                selection_strength=selection_strength
            )
            outputs.append(out_score)

        # -------------------
        # random-based
        # -------------------
        if N - num_score > 0:
            out_rand = self._reset_target_rand(rand_env_ids)
            outputs.append(out_rand)

        # -------------------
        # merge
        # -------------------
        return torch.cat(outputs, dim=0)