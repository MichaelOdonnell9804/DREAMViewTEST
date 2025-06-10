import ROOT
from types import SimpleNamespace
from event import Event
from geometry import Geometry, BOARD_TO_MODULES


class RDFEventProcessor:
    """Process events using ROOT.RDataFrame."""

    def __init__(self, rootfile: str, geometry: Geometry | None = None,
                 thresholds: dict | None = None,
                 order: str = Event.DEFAULT_ORDER):
        self.geometry = geometry or Geometry()
        self.thresholds = thresholds or {
            'c_offset': 0,
            's_offset': 0,
            'c_threshold': 0,
            's_threshold': 0,
            'nhitmin_c': 0,
            'nhitmin_s': 0,
        }
        self.order = order
        self.rdf = ROOT.RDataFrame("EventTree", rootfile)

        # Preload relevant columns
        self._board_branches = [f"FERS_Board{b}_energyHG" for b in BOARD_TO_MODULES]
        self._arrays = {
            name: self.rdf.Take["ROOT::VecOps::RVec<unsigned short>"](name).GetValue()
            for name in self._board_branches
        }
        self._runs = self.rdf.Take["unsigned int"]("run_n").GetValue()
        self._events = self.rdf.Take["unsigned int"]("event_n").GetValue()
        self._n = len(self._runs)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return self.events()

    def events(self):
        for i in range(self._n):
            entry = SimpleNamespace()
            for name in self._board_branches:
                setattr(entry, name, self._arrays[name][i])
            entry.run_n = int(self._runs[i])
            entry.event_n = int(self._events[i])
            yield Event.from_root_entry(entry, self.thresholds,
                                        self.geometry, order=self.order)
