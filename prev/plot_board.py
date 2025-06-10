import sys
import numpy as np
import matplotlib.pyplot as plt
import ROOT

# -----------------------------------------------------------------------------
# (A) find_index_for_event_n  (unchanged)
# -----------------------------------------------------------------------------
def find_index_for_event_n(rootfile, target_event_n, cycle=539):
    f = ROOT.TFile.Open(rootfile, "READ")
    if not f or f.IsZombie():
        print(f"ERROR: cannot open {rootfile!r}")
        return -1

    tree = f.Get(f"EventTree;{cycle}")
    if not tree:
        tree = f.Get("EventTree")
        if not tree:
            print("ERROR: TTree 'EventTree' not found")
            f.Close()
            return -1

    nentries = int(tree.GetEntries())
    for idx in range(nentries):
        tree.GetEntry(idx)
        evn = getattr(tree, "event_n", getattr(tree, "event", None))
        if evn == target_event_n:
            f.Close()
            return idx

    f.Close()
    return -1


# -----------------------------------------------------------------------------
# (B) load_scintillator_4x8  (modified to produce 8 rows × 4 columns)
# -----------------------------------------------------------------------------
def load_scintillator_4x8(rootfile, board, entry_idx, cycle=538):
    """
    Return an 8×4 array of scintillator energies for `board` at `entry_idx`.
    We assume the 64 channels form an 8×8 grid where every other column is Cherenkov.
    By taking arr2d[:, ::2], we drop the Cherenkov columns and keep 8×4 scintillator.
    """
    f = ROOT.TFile.Open(rootfile, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {rootfile!r}")

    tree = f.Get(f"EventTree;{cycle}")
    if not tree:
        tree = f.Get("EventTree")
        if not tree:
            f.Close()
            raise RuntimeError("TTree 'EventTree' not found")

    nentries = int(tree.GetEntries())
    if entry_idx < 0 or entry_idx >= nentries:
        f.Close()
        raise IndexError(f"entry_idx {entry_idx} out of range (0..{nentries-1})")

    tree.GetEntry(entry_idx)
    branch_name = f"FERS_Board{board}_energyHG"
    ener64 = getattr(tree, branch_name, None)
    if ener64 is None:
        f.Close()
        raise RuntimeError(f"Branch '{branch_name}' not found")

    arr64 = np.array([float(x) for x in ener64])
    if arr64.size != 64:
        f.Close()
        raise RuntimeError(
            f"Expected length 64 for {branch_name}, got {arr64.size}"
        )

    # Reshape into 8×8, then drop every other column → shape (8, 4)
    arr2d = arr64.reshape((8, 8))
    sc_2d = arr2d[:, ::2].copy()

    f.Close()
    return sc_2d


# -----------------------------------------------------------------------------
# (C) Build the full array (now 8 rows × (4×N_boards), with optional shifts)
# -----------------------------------------------------------------------------
def build_full_8x20_with_shifts(rootfile, event_n,
                                board_list=(4, 3, 2, 1, 0),
                                flips=None,
                                cycle=538):
    idx = find_index_for_event_n(rootfile, event_n, cycle=cycle)
    if idx < 0:
        raise RuntimeError(f"Could not find event_n = {event_n}")

    # (1) find the maximum downward shift requested
    max_shift = 0
    if flips:
        for b, (_, _, s) in flips.items():
            if s and s > max_shift:
                max_shift = s

    # (2) total_rows = 8 + max_shift (board height = 8)
    total_rows = 8 + max_shift
    total_cols = 4 * len(board_list)
    full_map = np.zeros((total_rows, total_cols), dtype=float)

    # (3) load each board’s 8×4 array, apply flips, then copy into full_map
    for slot, board in enumerate(board_list):
        subarr = load_scintillator_4x8(rootfile, board=board, entry_idx=idx, cycle=cycle)
        # subarr has shape (8,4)

        row_shift = 0
        if flips and (board in flips):
            do_flipud, do_fliplr, row_shift = flips[board]
            if do_flipud:
                subarr = np.flipud(subarr)
            if do_fliplr:
                subarr = np.fliplr(subarr)

        rstart = row_shift
        rend   = row_shift + 8   # subarr has 8 rows
        cstart = slot * 4
        cend   = cstart + 4

        full_map[rstart:rend, cstart:cend] = subarr

    return full_map


# -----------------------------------------------------------------------------
# (D) Plotting: mask zeros → make them white
# -----------------------------------------------------------------------------
def plot_full_8x20_with_shifts(rootfile, event_n, vmax=None,
                               flips=None, cycle=538,
                               board_list=(4, 3, 2, 1, 0)):

    full_map = build_full_8x20_with_shifts(rootfile, event_n,
                                           board_list=board_list,
                                           flips=flips,
                                           cycle=cycle)

    # Choose vmin/vmax (ignore zeros when computing max)
    vmin = 1.0
    if vmax is None:
        nonzero_max = full_map.max()  # any nonzero will be ≥ 1, so OK
        vmax = nonzero_max if nonzero_max > vmin else (vmin + 1.0)

    # ── Create a masked array that hides zeros ─────────────────────────────────
    masked_map = np.ma.masked_where(full_map == 0, full_map)

    # ── Copy and modify the colormap so that masked (i.e. zero) entries are white:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white", 1.0)   # “bad” ⇒ fully opaque white

    # ── Plot the masked array with that colormap ───────────────────────────────
    plt.figure(figsize=(10, 4))
    im = plt.imshow(masked_map, origin="upper",
                    cmap=cmap, vmin=vmin, vmax=vmax)

    # Add colorbar (covers only nonzero cells)
    cbar = plt.colorbar(im, label="energyHG (scintillator channels)")

    # Draw vertical red lines at every 4 columns (board boundaries)
    n_slots = full_map.shape[1] // 4
    for s in range(1, n_slots):
        plt.axvline(x=s * 4 - 0.5, color="red", linestyle="--", linewidth=1)

    # Annotate each nonzero cell with its integer value (in black)
    total_rows, total_cols = full_map.shape
    for r in range(total_rows):
        for c in range(total_cols):
            val = full_map[r, c]
            if val != 0:
                plt.text(c, r, f"{int(val)}",
                         color="black",
                         ha="center", va="center",
                         fontsize=6)

    plt.xlabel(f"combined column (0 … {total_cols - 1})")
    plt.ylabel(f"row (0 … {total_rows - 1})")
    plt.title(f"Full {total_rows}×{total_cols} face: Boards {board_list}, Event {event_n}\n"
              f"(flip/roll dict = {flips})")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# (E) Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python full_8x20_display.py /path/to/file.root event_n [vmax]")
        sys.exit(0)

    infile   = sys.argv[1]
    event_n  = int(sys.argv[2])
    vmax_val = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # Middle board (board 2) is shifted down by 4 rows:
    flips = {
        2: (False, False, 4),   # (flip_ud=False, flip_lr=False, row_shift=4)
    }

    plot_full_8x20_with_shifts(infile, event_n, vmax=vmax_val, flips=flips)
