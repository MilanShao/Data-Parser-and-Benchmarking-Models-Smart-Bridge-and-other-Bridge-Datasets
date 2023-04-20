import datetime
from typing import Optional, List, Literal, Generator, Dict, Any

from pbim_preprocessor.assembler.grandstand import GrandStandAssembler
from pbim_preprocessor.assembler.pbim import PBimAssembler
from pbim_preprocessor.assembler.z24 import Z24EMSAssembler, Z24PDTAssembler
from pbim_preprocessor.assembler.lux import LuxAssembler
from pbim_preprocessor.model import EOF


class AssemblerWrapper:
    def __init__(
        self,
        mode: str,
        base_assembler: PBimAssembler
        | GrandStandAssembler
        | Z24EMSAssembler
        | Z24PDTAssembler
        | LuxAssembler,
    ):
        self._mode = mode
        self._base_assembler = base_assembler

    def assemble(
        self,
        start_time: Optional[datetime.datetime],
        end_time: Optional[datetime.datetime],
        scenario: Optional[str],
        channels: Optional[List[str]],
        scenario_type: Optional[Literal["avt", "fvt"]],
    ) -> Generator[Dict[str, float] | EOF, Any, None]:
        match self._mode:
            case "pbim":
                yield from self._base_assembler.assemble(start_time, end_time, channels)
            case "grandstand":
                yield from self._base_assembler.assemble(scenario, channels)
            case "z24-ems":
                yield from self._base_assembler.assemble(start_time, end_time, channels)
            case "z24-pdt":
                yield from self._base_assembler.assemble(
                    int(scenario), scenario_type, channels
                )
            case "lux":
                yield from self._base_assembler.assemble(start_time, end_time, channels)
            case _:
                raise ValueError(f"Unknown mode {self._mode}")
