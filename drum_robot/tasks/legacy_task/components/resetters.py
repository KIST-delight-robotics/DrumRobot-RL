
"""
리셋 클래스 

legacy task 를 위해 남겨놓은 파일
"""

from __future__ import annotations

from dataclasses import dataclass
from mido import MidiFile
import torch
from pathlib import Path
import random
import time
import math

"""
RDS 리셋
"""

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
class RdsGeneratorCfg:

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

class RdsGenerator:

    def __init__(self, num_envs: int, device: torch.device | str, cfg: RdsGeneratorCfg):
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

    def reset_target(self, env_ids, temperature=0.5):
        weights = self.score.clone()

        # temperature 랜덤성 조절
        weights = torch.pow(weights, temperature)

        probs = weights / (weights.sum() + 1e-6)

        # probs를 기반으로 인덱스를 랜덤 샘플링
        idx = torch.multinomial(
            probs,
            num_samples=len(env_ids),
            replacement=True    # 같은 인덱스가 여러 번 뽑힐 수 있음
        )

        robotic_drum_score = self.rds[idx]
        return robotic_drum_score

    def reset_target_rand(self, env_ids):
        N = len(env_ids)
        T = self.episode_length
        M = self.cfg.num_drum

        robotic_drum_score = torch.zeros((N, T, M), device=self.device, dtype=torch.int64)

        s = self.cfg.start_offset_steps
        e = T - self.cfg.hit_window_step

        k = 10
        for i in range(k):
            si = (int)(s + (2 * i) * (e - s) / (2 * k - 1))
            ei = (int)(s + (2 * i + 1) * (e - s) / (2 * k - 1))
            time_rand = torch.randint(si, ei, (N,), device=self.device)
            inst_rand = torch.randint(0, M, (N,), device=self.device)

            robotic_drum_score[torch.arange(N), time_rand, inst_rand] = 1

        return robotic_drum_score


"""
init pos 리셋
"""

@dataclass
class PartLength:
    upper_arm: float = 0.2303
    lower_arm: float = 0.200
    stick: float = 0.325 + 0.048
    waist: float = 0.520

class IKSolver:
    def __init__(self, part: PartLength, device: torch.device | str):

        self.device = torch.device(device)

        self.upper_arm = torch.tensor(part.upper_arm, device=self.device, dtype=torch.float32)
        self.lower_arm = torch.tensor(part.lower_arm, device=self.device, dtype=torch.float32)
        self.stick = torch.tensor(part.stick, device=self.device, dtype=torch.float32)
        self.waist = torch.tensor(part.waist, device=self.device, dtype=torch.float32)

    def _get_length(self, theta: torch.Tensor) -> torch.Tensor:

        x = self.lower_arm + self.stick * torch.cos(theta)
        y = self.stick * torch.sin(theta)

        return torch.sqrt(x * x + y * y)

    def _get_theta(self, theta: torch.Tensor) -> torch.Tensor:

        x = self.lower_arm + self.stick * torch.cos(theta)
        y = self.stick * torch.sin(theta)

        return torch.atan2(y, x)

    @torch.no_grad()
    def solve_geometric_ik(
        self,
        pR: torch.Tensor,       # (N, 3)
        pL: torch.Tensor,       # (N, 3)
        theta0: torch.Tensor,   # (N, 1)
        theta7: torch.Tensor,   # (N, 1)
        theta8: torch.Tensor,   # (N, 1)
    ) -> torch.Tensor:

        err = torch.zeros_like(theta0)

        XR, YR, ZR = pR[:, 0], pR[:, 1], pR[:, 2]
        XL, YL, ZL = pL[:, 0], pL[:, 1], pL[:, 2]

        # constants
        L1 = self.upper_arm
        S  = self.waist

        L2_R = self._get_length(theta7)
        L2_L = self._get_length(theta8)

        # shoulder positions
        shoulderXR = 0.5 * S * torch.cos(theta0)
        shoulderYR = 0.5 * S * torch.sin(theta0)
        shoulderXL = -0.5 * S * torch.cos(theta0)
        shoulderYL = -0.5 * S * torch.sin(theta0)

        # ---- q1 ----
        theta01 = torch.atan2(YR - shoulderYR, XR - shoulderXR)
        theta1 = theta01 - theta0

        # theta1 range: 0 ~ 150deg
        bad1 = (theta1 < 0.0) | (theta1 > 150.0 * math.pi / 180.0)
        err = torch.where(bad1, torch.ones_like(err), err)

        # ---- q2 ----
        theta02 = torch.atan2(YL - shoulderYL, XL - shoulderXL)
        theta2 = theta02 - theta0

        # theta2 range: 30deg ~ 180deg
        bad2 = (theta2 < 30.0 * math.pi / 180.0) | (theta2 > math.pi)
        err = torch.where(bad2, torch.ones_like(err), err)

        # =========================
        # Right arm geometry
        # =========================
        zeta_r = - 1 * ZR
        r2_r = (YR - shoulderYR) ** 2 + (XR - shoulderXR) ** 2

        x_r = zeta_r * zeta_r + r2_r - L1 * L1 - L2_R * L2_R
        rad_r = 4.0 * L1 * L1 * L2_R * L2_R - x_r * x_r

        # C++: rad<0 이면 즉시 return (theta0=99, err=1)
        sqrt_bad_r = rad_r < 0.0
        err = torch.where(sqrt_bad_r, torch.ones_like(err), err)

        # 안전 sqrt
        y_r = torch.sqrt(torch.clamp(rad_r, min=0.0))

        theta4 = torch.atan2(y_r, x_r)
        theta34 = torch.atan2(torch.sqrt(torch.clamp(r2_r, min=0.0)), zeta_r)
        theta3 = theta34 - torch.atan2(L2_R * torch.sin(theta4), L1 + L2_R * torch.cos(theta4))

        # theta3 range: -45 ~ 90deg
        bad3 = (theta3 < -45.0 * math.pi / 180.0) | (theta3 > 90.0 * math.pi / 180.0)
        err = torch.where(bad3, torch.ones_like(err), err)

        # =========================
        # Left arm geometry
        # =========================
        zeta_l = -1 * ZL
        r2_l = (YL - shoulderYL) ** 2 + (XL - shoulderXL) ** 2

        x_l = zeta_l * zeta_l + r2_l - L1 * L1 - L2_L * L2_L
        rad_l = 4.0 * L1 * L1 * L2_L * L2_L - x_l * x_l

        sqrt_bad_l = rad_l < 0.0
        err = torch.where(sqrt_bad_l, torch.ones_like(err), err)

        y_l = torch.sqrt(torch.clamp(rad_l, min=0.0))

        theta6 = torch.atan2(y_l, x_l)
        theta56 = torch.atan2(torch.sqrt(torch.clamp(r2_l, min=0.0)), zeta_l)
        theta5 = theta56 - torch.atan2(L2_L * torch.sin(theta6), L1 + L2_L * torch.cos(theta6))

        # theta5 range: -45 ~ 90deg
        bad5 = (theta5 < -45.0 * math.pi / 180.0) | (theta5 > 90.0 * math.pi / 180.0)
        err = torch.where(bad5, torch.ones_like(err), err)

        # adjust theta4/theta6 by stick geometry
        theta4 = theta4 - self._get_theta(theta7)
        theta6 = theta6 - self._get_theta(theta8)

        # theta4 range: 0 ~ 140deg
        bad4 = (theta4 < 0.0) | (theta4 > 140.0 * math.pi / 180.0)
        err = torch.where(bad4, torch.ones_like(err), err)

        # theta6 range: 0 ~ 140deg
        bad6 = (theta6 < 0.0) | (theta6 > 140.0 * math.pi / 180.0)
        err = torch.where(bad6, torch.ones_like(err), err)

        out = torch.stack(
            [theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, err],
            dim=-1
        )  # (N,10)

        # nan 체크
        nan_bad = torch.isnan(out[:, :9]).any(dim=-1)
        out[nan_bad, 9] = 1.0

        # C++의 "sqrt 음수면 즉시 return theta0=99" 동작을 배치로 반영
        # (우측/좌측 중 하나라도 sqrt_bad면 해당 row를 에러상태로 강제)
        sqrt_bad = sqrt_bad_r | sqrt_bad_l
        if sqrt_bad.any():
            out[sqrt_bad, :] = 0.0
            out[sqrt_bad, 0] = 99.0
            out[sqrt_bad, 9] = 1.0

        return out

@dataclass
class PosGeneratorCfg:
    # 제어하는 관절 개수
    num_ctrl_joint: int = 9

    # 랜덤 초기 관절각 범위
    init_joint_range = {
        "waist_joint":          (-30*math.pi/180,   30*math.pi/180),
        "left_shoulder_1":      ( 60*math.pi/180,   120*math.pi/180),
        "left_shoulder_2":      (-40*math.pi/180,   50*math.pi/180),
        "left_elbow":           ( 60*math.pi/180,   120*math.pi/180),
        "right_shoulder_1":     ( 60*math.pi/180,   120*math.pi/180),
        "right_shoulder_2":     (-40*math.pi/180,   50*math.pi/180),
        "right_elbow":          ( 60*math.pi/180,   120*math.pi/180),
        "left_wrist":           ( 20*math.pi/180,   50*math.pi/180),
        "right_wrist":          ( 20*math.pi/180,   50*math.pi/180),
    }

    # 양 팔이 위치 가능한 드럼 조합 [L, R]
    drum_pairs = [
        (1, 1), (5, 1),
        (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (8, 2),
        (1, 3), (3, 3), (4, 3), (5, 3), (8, 3),
        (1, 4), (4, 4), (5, 4), (8, 4),
        (1, 5), (5, 5),
        (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (8, 6),
        (1, 7), (2, 7), (3, 7), (4, 7), (6, 7), (7, 7),
        (1, 8), (4, 8), (5, 8), (8, 8),
    ]

    height_above_drum: float = 0.1

    inst_name_to_idx = {
        "waist_joint":          0,
        "left_shoulder_1":      2,
        "left_shoulder_2":      5,
        "left_elbow":           6,
        "right_shoulder_1":     1,
        "right_shoulder_2":     3,
        "right_elbow":          4,
        "left_wrist":           8,
        "right_wrist":          7,
    }

    joint_noise_scale: float = 5*math.pi/180

class PosGenerator:
    def __init__(
            self,
            device: torch.device | str,
            cfg: PosGeneratorCfg,
            ctrl_joint_names: list,
            instruments: dict,
    ):
        self.device = torch.device(device)
        self.cfg = cfg
        self.ctrl_joint_names = ctrl_joint_names
        self.instruments = instruments

        self._init_rand()
        self._init_ik()

    def _init_rand(self):
        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """
        self.init_joint_min = torch.tensor(
            [self.cfg.init_joint_range[name][0] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

        self.init_joint_max = torch.tensor(
            [self.cfg.init_joint_range[name][1] for name in self.ctrl_joint_names],
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

    def _init_ik(self):
        N = len(self.cfg.drum_pairs)
        ik_solver = IKSolver(part=PartLength(), device=self.device)

        inst_pos = torch.tensor(
            list(self.instruments.values()),
            device=self.device,
            dtype=torch.float32
        )   # (8, 3)

        drum_pairs = torch.tensor(
            self.cfg.drum_pairs,
            device=self.device,
            dtype=torch.int32
        )   # (N, 2)
        drum_pairs_idx = drum_pairs - 1

        p = inst_pos[drum_pairs_idx, :]  # (N, 2, 3)
        pl = p[:, 0, :]     # (N, 3)
        pr = p[:, 1, :]

        pl[:, 2] = pl[:, 2] + self.cfg.height_above_drum
        pr[:, 2] = pr[:, 2] + self.cfg.height_above_drum

        pm_xy = (pl[:, 0:2] + pr[:, 0:2]) / 2
        the0 = torch.atan2(pm_xy[:, 1], pm_xy[:, 0]) - 90*math.pi/180

        the7 = torch.full((N,), 25*math.pi/180, device=self.device)
        the8 = torch.full((N,), 25*math.pi/180, device=self.device)

        out = ik_solver.solve_geometric_ik(pr, pl, the0, the7, the8)    # (N, 10)

        self.pos_angle = torch.zeros((N, self.cfg.num_ctrl_joint), device=self.device)

        """ 반드시 self.ctrl_joint_names 순서대로 텐서를 만들어여 함 """
        for i in range(self.cfg.num_ctrl_joint):
            name = self.ctrl_joint_names[i]
            idx = self.cfg.inst_name_to_idx[name]
            self.pos_angle[:, i] = out[:, idx]

    def reset_init_pos_rand(self, env_ids):
        # 랜덤 각도 초기 위치
        rand = torch.rand((len(env_ids), self.cfg.num_ctrl_joint), device=self.device)   # (N, 9)
        init_pos = self.init_joint_min + rand * (self.init_joint_max - self.init_joint_min)

        return init_pos
    
    def reset_init_pos(self, env_ids):
        N = len(self.cfg.drum_pairs)

        rand = torch.randint(0, N, (len(env_ids),), device=self.device)

        init_pos = self.pos_angle[rand, :]
        init_pos = init_pos + self.cfg.joint_noise_scale * torch.randn_like(init_pos)

        return init_pos