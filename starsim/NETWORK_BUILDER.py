import numpy as np 
import pandas as pd 
from typing import Tuple
import sys
from pathlib import Path

import io
from matplotlib import pyplot as plt
import imageio.v2 as imageio

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))
import starsim as ss 


def read_data(data,):
    df = pd.read_csv(data, index_col=0)
    return df 

def clean_data(df: pd.DataFrame, beta: float = 1.0) -> pd.DataFrame:
    df = df[df["type"] == "contact"].copy()
    df = df.dropna(subset=["user_id", "peer_id", "time"])
    df = df.drop_duplicates(subset=["user_id", "peer_id", "time"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["beta"] = beta
    return df

def remap_ids(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    all_ids = pd.Index(pd.unique(pd.concat([df["user_id"], df["peer_id"]], ignore_index=True)))
    id_map = {old: new for new, old in enumerate(all_ids)}

    out = df.copy()
    out["p1"] = out["user_id"].map(id_map).astype("int64")
    out["p2"] = out["peer_id"].map(id_map).astype("int64")
    return out, len(all_ids)

def add_subdaily_timeline(
    df: pd.DataFrame,
    time_col: str = "time",
    duration_col: str = "contact_length",
    tick: pd.Timedelta = pd.Timedelta(milliseconds=1),
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Convert timestamps to a fixed subdaily simulation timeline.

    The default tick is 1 ms (thousandths of a second), which preserves the
    full resolution in contact timestamps.

    Assumes `duration_col` is stored in milliseconds.
    """
    out = df.copy()
    start_date = out[time_col].min().floor(tick)
    stop_date = out[time_col].max().ceil(tick) + tick

    out["start_offset_ti"] = ((out[time_col] - start_date) / tick).round().astype("int64")
    tick_ms = tick / pd.Timedelta(milliseconds=1)
    out["dur_ti"] = np.ceil(out[duration_col].fillna(0) / tick_ms).astype("int64").clip(lower=1)

    return out, start_date, stop_date

class EpigamesNet(ss.DynamicNetwork):
    def __init__(self, df, start_col='start_offset_ti', dur_col='dur_ti', **kwargs):
        super().__init__(**kwargs)
        self.df = df.sort_values(start_col).reset_index(drop=True)
        self.start_col = start_col
        self.dur_col = dur_col 
        self._next = 0

    def init_pre(self, sim): 
        super().init_pre(sim)
        self._starts = self.df[self.start_col].to_numpy()
        return
    
    def add_pairs(self):
        added = 0
        while self._next < len(self.df) and self._starts[self._next] == self.ti:
            row = self.df.iloc[self._next]

            if added < 5:
                print(f"[ti={self.ti}] adding edge ({row.p1}, {row.p2}) for {row[self.dur_col]} ms")

            self.append(
                p1=np.array([int(row.p1)], dtype=ss.dtypes.int),
                p2=np.array([int(row.p2)], dtype=ss.dtypes.int),
                beta=np.array([float(row.beta)], dtype=ss.dtypes.float),
                dur=np.array([float(row[self.dur_col])], dtype=ss.dtypes.float),
            )
            self._next += 1
            added += 1

        return added


class SyntheticEpigameNet(ss.DynamicNetwork):
    """
    Synthetic daily contact network for Epigames-like simulations.

    Contacts are generated endogenously each timestep from active agents.
    Quarantined agents have reduced contact rates.
    """

    def __init__(
        self,
        beta=1.0,
        mean_contacts=4.0,
        quarantine_multiplier=0.25,
        dur_days=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta0 = beta
        self.mean_contacts = mean_contacts
        self.quarantine_multiplier = quarantine_multiplier
        self.dur_days = dur_days

    def init_post(self):
        super().init_post()
        self._rng = self.sim.rng  # Starsim RNG
        self._added_today = 0

    def _get_active_uids(self):
        # Default: all alive agents
        uids = self.sim.people.uid[self.sim.people.alive]
        return np.asarray(uids, dtype=ss.dtypes.int)

    def _get_contact_rate(self, uids):
        """
        Reduce contacts for quarantined people if the state exists.
        Adjust this logic to match your actual ABM state name.
        """
        rate = np.full(len(uids), self.mean_contacts, dtype=float)

        # Try common quarantine state names; adapt if your project uses something else
        people = self.sim.people
        if hasattr(people, "quarantine"):
            q = np.asarray(people.quarantine[uids]).astype(bool)
            rate[q] *= self.quarantine_multiplier

        return rate

    def add_pairs(self):
        active = self._get_active_uids()
        if len(active) < 2:
            return 0

        rates = self._get_contact_rate(active)
        n_contacts = self._rng.poisson(rates).astype(int)

        # Total desired contacts for the day
        total_half_edges = int(n_contacts.sum())
        if total_half_edges < 2:
            return 0

        # Expand each uid according to its contact demand
        p1 = np.repeat(active, n_contacts)

        # Randomly permute partners among the same active pool
        p2 = self._rng.choice(active, size=len(p1), replace=True)

        # Remove self-contacts
        mask = p1 != p2
        p1 = p1[mask]
        p2 = p2[mask]

        if len(p1) == 0:
            return 0

        beta = np.full(len(p1), self.beta0, dtype=float)
        dur = np.full(len(p1), self.dur_days, dtype=float)

        self.append(
            p1=p1.astype(ss.dtypes.int),
            p2=p2.astype(ss.dtypes.int),
            beta=beta.astype(ss.dtypes.float),
            dur=dur.astype(ss.dtypes.float),
        )

        return len(p1)

    def remove_pairs(self):
        # If you use dur, Starsim will naturally expire edges over time.
        return 0
    

def build_network(csv_path: str | Path, label: str = "Epigames"):
    df = read_data(csv_path)
    df = clean_data(df)
    df, n_agents = remap_ids(df)
    df, start_date, stop_date = add_subdaily_timeline(df)
    net = EpigamesNet(df, label=label)
    start_date = ss.date(start_date)
    stop_date = ss.date(stop_date)
    return net, n_agents, start_date, stop_date

def build_network_synth(csv_path: str | Path, label: str = "Epigames_Synth"):
    df = read_data(csv_path)
    df = clean_data(df)
    df, n_agents = remap_ids(df)
    df, start_date, stop_date = add_subdaily_timeline(df)
    net = SyntheticEpigameNet(df, label=label)
    start_date = ss.date(start_date)
    stop_date = ss.date(stop_date)
    return net, n_agents, start_date, stop_date


# === GIF helpers ===
def _to_pandas_timestamp(value) -> pd.Timestamp:
    if hasattr(value, "to_pandas"):
        return pd.Timestamp(value.to_pandas())
    return pd.Timestamp(value)


def _circle_layout(n_agents: int, radius: float = 1.0) -> np.ndarray:
    if n_agents <= 0:
        return np.zeros((0, 2), dtype=float)
    angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
    return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))


def _active_edges_at_ti(df: pd.DataFrame, ti: int, start_col: str = "start_offset_ti", dur_col: str = "dur_ti") -> np.ndarray:
    active = df[(df[start_col] <= ti) & (ti < (df[start_col] + df[dur_col]))]
    if active.empty:
        return np.empty((0, 2), dtype=int)
    return active[["p1", "p2"]].to_numpy(dtype=int)


def _render_network_frame(
    positions: np.ndarray,
    edges: np.ndarray,
    infected: np.ndarray | None,
    title: str,
    subtitle: str | None = None,
    dpi: int = 130,
    edge_color: str = "0.75",
    edge_alpha: float = 0.35,
    edge_linewidth: float = 0.8,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(7, 7))

    for p1, p2 in edges:
        x0, y0 = positions[p1]
        x1, y1 = positions[p2]
        ax.plot([x0, x1], [y0, y1], color=edge_color, alpha=edge_alpha, linewidth=edge_linewidth, zorder=1)

    if infected is None:
        node_colors = np.full(len(positions), "tab:blue", dtype=object)
    else:
        node_colors = np.where(np.asarray(infected, dtype=bool), "tab:red", "tab:blue")

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_colors,
        s=35,
        edgecolors="white",
        linewidths=0.5,
        zorder=2,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    if subtitle:
        ax.text(0.5, 0.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def save_network_gif(
    df: pd.DataFrame,
    n_agents: int,
    start_date,
    out_path: str | Path = "network_over_time.gif",
    frame_stride_ti: int | None = None,
    max_frames: int = 240,
    frame_duration: float = 0.08,
) -> Path:
    """Create a GIF showing the active network over time.

    This uses the contact table directly and does not require running the sim.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    positions = _circle_layout(n_agents)
    end_ti = int((df["start_offset_ti"] + df["dur_ti"]).max())
    end_ti = max(end_ti, 0)

    if frame_stride_ti is None:
        n_frames = min(max_frames, end_ti + 1 if end_ti > 0 else 1)
        sample_tis = np.unique(np.linspace(0, end_ti, num=n_frames, dtype=int))
    else:
        sample_tis = np.arange(0, end_ti + 1, frame_stride_ti, dtype=int)
        if sample_tis.size == 0:
            sample_tis = np.array([0], dtype=int)

    base_time = _to_pandas_timestamp(start_date)
    frames = []
    for ti in sample_tis:
        edges = _active_edges_at_ti(df, int(ti))
        frame_time = base_time + pd.Timedelta(milliseconds=int(ti))
        title = f"Active network at ti={int(ti)}"
        subtitle = f"{frame_time} | edges={len(edges)}"
        frames.append(_render_network_frame(positions, edges, None, title=title, subtitle=subtitle))

    imageio.mimsave(out_path, frames, duration=frame_duration)
    return out_path


def save_infection_gif(
    csv_path: str | Path,
    out_path: str | Path = "infection_over_time.gif",
    target_ms: int = 130000,
    disease_name: str = "sis",
    init_prev: float = 0.1,
    frame_stride_ti: int | None = None,
    max_frames: int = 240,
    frame_duration: float = 0.08,
) -> Path:
    """Create a GIF showing the network and infection state over time.

    This builds a fresh simulation so it can record infection state over time.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net, n_agents, start_date, _ = build_network(csv_path)
    df = net.df
    ms = ss.days(1 / 86_400_000)

    sim = ss.Sim(
        n_agents=n_agents,
        networks=[net],
        diseases=ss.SIS(init_prev=init_prev),
        start=start_date,
        stop=start_date + pd.Timedelta(milliseconds=target_ms),
        dt=ms,
    )
    sim.init()

    positions = _circle_layout(n_agents)
    end_ti = max(int(target_ms) - 1, 0)
    if frame_stride_ti is None:
        n_frames = min(max_frames, end_ti + 1 if end_ti > 0 else 1)
        sample_tis = np.unique(np.linspace(0, end_ti, num=n_frames, dtype=int))
    else:
        sample_tis = np.arange(0, end_ti + 1, frame_stride_ti, dtype=int)
        if sample_tis.size == 0:
            sample_tis = np.array([0], dtype=int)

    sample_set = {int(x) for x in sample_tis}
    frames = []

    def capture_frame(current_ti: int):
        disease = getattr(sim.people, disease_name)
        infected = np.asarray(disease.infected, dtype=bool)
        edges = _active_edges_at_ti(df, current_ti)
        frame_time = _to_pandas_timestamp(start_date) + pd.Timedelta(milliseconds=int(current_ti))
        title = f"Network + infection at ti={int(current_ti)}"
        subtitle = f"{frame_time} | infected={int(infected.sum())}/{len(infected)} | edges={len(edges)}"
        frames.append(
            _render_network_frame(
                positions,
                edges,
                infected,
                title=title,
                subtitle=subtitle,
                edge_color="0.55",
                edge_alpha=0.55,
                edge_linewidth=1.1,
            )
        )

    if 0 in sample_set:
        capture_frame(0)

    while not sim.t.finished:
        sim.run_one_step()
        current_ti = int(getattr(sim.t, "ti", 0))
        if current_ti in sample_set:
            capture_frame(current_ti)

    imageio.mimsave(out_path, frames, duration=frame_duration)
    return out_path

def agent_prevalence_at_ti(
    sim,
    ti: int | None = None,
    disease_name: str = "sis",
    # NOTE: we only have one network so we probably don't need this nomenclature, but keep for the future?
    network_idx: int = 0,
) -> pd.DataFrame:
    """Return agent-level infection status and contact-conditioned prevalence.

    If `ti` is provided, the function checks that the simulation is currently
    at that timestep before returning the report.

    The returned `contact_infected_fraction` is computed over each agent's
    currently active contacts in the selected network, not over the whole
    population.
    """
    current_ti = getattr(sim, "ti", None)
    if current_ti is None and hasattr(sim, "t"):
        current_ti = getattr(sim.t, "ti", None)

    if ti is not None and current_ti != ti:
        raise ValueError(f"Simulation is at ti={current_ti}, not ti={ti}.")

    disease = getattr(sim.people, disease_name)
    infected = np.asarray(disease.infected, dtype=bool)
    n_agents = len(infected)

    network = sim.networks[network_idx]
    contact_counts = np.zeros(n_agents, dtype=int)
    contact_infected_fraction = np.full(n_agents, np.nan, dtype=float)

    for agent_id in range(n_agents):
        #NOTE: One caveat: because find_contacts(...) returns a set/unique contact list, this treats repeated connections to the same person as one contact in the fraction. 
        # Starsim notes that find_contacts is intended for contact queries and not cases where multiple connections should count differently.
        # i.e. if people are connected multple times across dt they will only be shown once
        # since dt = 1 milisecond in this case it's probably ok
        contacts = network.find_contacts(np.array([agent_id], dtype=ss.dtypes.int), as_array=True)
        contacts = np.asarray(contacts, dtype=int)
        contacts = contacts[contacts != agent_id]

        contact_counts[agent_id] = int(len(contacts))
        if contacts.size:
            contact_infected_fraction[agent_id] = float(infected[contacts].mean())

    out = pd.DataFrame({
        "agent_id": np.arange(n_agents, dtype=int),
        "infected": infected,
        "contact_count": contact_counts,
        "contact_infected_fraction": contact_infected_fraction,
        "ti": current_ti,
    })

    try:
        out["time"] = sim.t.now("date")
    except Exception:
        pass

    return out

if __name__ == "__main__":
    csv_path = "data_ingestion/histories.csv"

    
    
    
    
    
    # Load and process data step by step for inspection
    df = read_data(csv_path)
    df = clean_data(df)
    df, n_agents = remap_ids(df)
    df, start_date, stop_date = add_subdaily_timeline(df)

    print("\n=== BASIC INFO ===")
    print(f"Number of agents: {n_agents}")
    print(f"Start date: {start_date}")
    print(f"Stop date: {stop_date}")

    print("\n=== SAMPLE OF PROCESSED DATA ===")
    print(df[["time", "start_offset_ti", "dur_ti"]].head(10))

    # Check timestep differences (should reflect ms resolution)
    diffs = df["start_offset_ti"].sort_values().diff().dropna()
    print("\n=== TIMESTEP DIFFERENCES ===")
    print("Min diff (ti units):", diffs.min())
    print("Max diff (ti units):", diffs.max())

    # Convert a few back to real time to sanity check
    print("\n=== RECONSTRUCTED TIMES (sanity check) ===")
    tick = pd.Timedelta(milliseconds=1)
    for i in range(5):
        row = df.iloc[i]
        reconstructed = start_date + row["start_offset_ti"] * tick
        print(f"Original: {row['time']} | Reconstructed: {reconstructed}")

    # Check duration interpretation
    print("\n=== DURATION CHECK ===")
    for i in range(5):
        row = df.iloc[i]
        duration_ms = row["dur_ti"]  # since 1 tick = 1 ms
        print(f"dur_ti: {row['dur_ti']} -> approx duration (ms): {duration_ms}")

    print("\n=== FIRST 10 START TIMES ===")
    print(df["start_offset_ti"].sort_values().head(10))

    print("\n=== RUNNING SMALL SIM ===")

    net, n_agents, start_date, stop_date = build_network(csv_path)

    ms = ss.days(1/86_400_000)

    target_ms = 13000

    sim = ss.Sim(
        n_agents=n_agents,
        networks=[net],
        diseases = ss.SIS(init_prev = 0.1),
        start=start_date,
        stop=start_date + pd.Timedelta(milliseconds=target_ms),
        dt=ms,
    )

    sim.run()

    print("\n=== AGENT-LEVEL CONTACT PREVALENCE AT TARGET TIMESTEP ===")
    report = agent_prevalence_at_ti(sim, ti=target_ms, disease_name="sis")
    print(report.head(20))
    print("\nAgents with at least one contact:")
    print(report.loc[report["contact_count"] > 0, ["agent_id", "contact_count", "contact_infected_fraction", "infected"]].head(20))
    print("\nAgents whose current contacts are infected:")
    print(report.loc[report["contact_infected_fraction"] > 0, ["agent_id", "contact_count", "contact_infected_fraction"]].head(20))

    print("\n=== SAVING NETWORK GIF ===")
    network_gif_path = save_network_gif(
        df=df,
        n_agents=n_agents,
        start_date=start_date,
        out_path=Path("data_ingestion/network_over_time.gif"),
    )
    print(f"Saved network GIF to {network_gif_path}")

    print("\n=== SAVING INFECTION GIF ===")
    infection_gif_path = save_infection_gif(
        csv_path=csv_path,
        out_path=Path("data_ingestion/infection_over_time.gif"),
        target_ms=target_ms,
        disease_name="sis",
        init_prev=0.1,
    )
    print(f"Saved infection GIF to {infection_gif_path}")

    print("\n=== DONE ===")