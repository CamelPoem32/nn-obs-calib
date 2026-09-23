'''Deterministic parameter metadata for calibration observability computations.'''

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from obscalib.calibration.context import CALIBRATION_CONTEXT_DIM


ROTATION_PARAMETER_NAMES = (
    'phi_x',
    'phi_y',
    'phi_z',
)

TRANSLATION_PARAMETER_NAMES = (
    'rho_x',
    'rho_y',
    'rho_z',
)

TIME_OFFSET_PARAMETER_NAMES = (
    'tau',
)

CALIBRATION_PARAMETER_NAMES = (
    ROTATION_PARAMETER_NAMES
    + TRANSLATION_PARAMETER_NAMES
    + TIME_OFFSET_PARAMETER_NAMES
)

ROTATION_DIM = len(ROTATION_PARAMETER_NAMES)
TRANSLATION_DIM = len(TRANSLATION_PARAMETER_NAMES)
SPATIAL_CALIBRATION_DIM = ROTATION_DIM + TRANSLATION_DIM


if len(CALIBRATION_PARAMETER_NAMES) != CALIBRATION_CONTEXT_DIM:
    raise RuntimeError(
        'Calibration parameter metadata is inconsistent with '
        'CALIBRATION_CONTEXT_DIM: '
        f'{len(CALIBRATION_PARAMETER_NAMES)} names for dimension '
        f'{CALIBRATION_CONTEXT_DIM}.'
    )


@dataclass(frozen=True)
class CalibrationParameterBlock:
    '''Contiguous [phi, rho, tau] columns for one calibration key.'''

    calibration_key: str
    start: int

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_key, str) or not self.calibration_key:
            raise ValueError('calibration_key must be a non-empty string.')

        if self.start < 0:
            raise ValueError('start must be nonnegative.')

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return CALIBRATION_PARAMETER_NAMES

    @property
    def dimension(self) -> int:
        return CALIBRATION_CONTEXT_DIM

    @property
    def stop(self) -> int:
        return self.start + self.dimension

    @property
    def parameter_slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def rotation_slice(self) -> slice:
        '''Columns for the SO(3) perturbation phi.'''

        return slice(
            self.start,
            self.start + ROTATION_DIM,
        )

    @property
    def translation_slice(self) -> slice:
        '''Columns for the translational perturbation rho.'''

        translation_start = self.start + ROTATION_DIM

        return slice(
            translation_start,
            translation_start + TRANSLATION_DIM,
        )

    @property
    def spatial_slice(self) -> slice:
        '''Columns for xi = [phi, rho].'''

        return slice(
            self.start,
            self.start + SPATIAL_CALIBRATION_DIM,
        )

    @property
    def time_offset_slice(self) -> slice:
        '''Column for additive temporal calibration tau.'''

        time_offset_start = self.start + SPATIAL_CALIBRATION_DIM

        return slice(
            time_offset_start,
            time_offset_start + len(TIME_OFFSET_PARAMETER_NAMES),
        )

    def parameter_index(
        self,
        parameter_name: str,
    ) -> int:
        '''Return the global column index of one calibration parameter.'''

        try:
            local_index = self.parameter_names.index(parameter_name)
        except ValueError as exc:
            raise KeyError(
                f'Unknown calibration parameter {parameter_name!r} '
                f'for key {self.calibration_key!r}.'
            ) from exc

        return self.start + local_index


@dataclass(frozen=True)
class CalibrationParameterLayout:
    '''Explicit ordering of calibration Fisher-matrix rows and columns.

    The caller supplies calibration keys in the desired scientific order.
    Mapping iteration order is deliberately not used to define the matrix
    layout.

    Every current CalibrationState contributes seven parameters ordered as

        [phi_x, phi_y, phi_z, rho_x, rho_y, rho_z, tau]

    reusing the calibration_state_to_context convention.

    This layout contains only variables that remain in the final calibration
    Fisher matrix. Trajectory states and other nuisance variables are
    represented separately and are marginalized before the final matrix is
    formed.
    '''

    blocks: tuple[CalibrationParameterBlock, ...]

    def __post_init__(self) -> None:
        calibration_keys = tuple(
            block.calibration_key
            for block in self.blocks
        )

        if len(calibration_keys) != len(set(calibration_keys)):
            raise ValueError('Calibration layout keys must be unique.')

        expected_start = 0

        for block in self.blocks:
            if block.start != expected_start:
                raise ValueError(
                    'Calibration layout blocks must be contiguous '
                    'and start at zero.'
                )

            expected_start = block.stop

    @classmethod
    def from_calibration_keys(
        cls,
        calibration_keys: Sequence[str],
    ) -> 'CalibrationParameterLayout':
        '''Build a layout from an explicitly ordered sequence of keys.'''

        if isinstance(calibration_keys, str):
            raise TypeError(
                'calibration_keys must be a sequence of keys, '
                'not one string.'
            )

        calibration_keys = tuple(calibration_keys)

        if not calibration_keys:
            raise ValueError(
                'At least one calibration key is required.'
            )

        if any(
            not isinstance(calibration_key, str)
            or not calibration_key
            for calibration_key in calibration_keys
        ):
            raise ValueError(
                'Calibration keys must be non-empty strings.'
            )

        if len(calibration_keys) != len(set(calibration_keys)):
            raise ValueError(
                'Calibration keys must be unique.'
            )

        return cls(
            blocks=tuple(
                CalibrationParameterBlock(
                    calibration_key=calibration_key,
                    start=block_index * CALIBRATION_CONTEXT_DIM,
                )
                for block_index, calibration_key
                in enumerate(calibration_keys)
            )
        )

    @property
    def calibration_keys(self) -> tuple[str, ...]:
        return tuple(
            block.calibration_key
            for block in self.blocks
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        '''Return qualified parameter names in Fisher-matrix order.'''

        return tuple(
            f'{block.calibration_key}.{parameter_name}'
            for block in self.blocks
            for parameter_name in block.parameter_names
        )

    @property
    def total_dimension(self) -> int:
        return sum(
            block.dimension
            for block in self.blocks
        )

    def block_for(
        self,
        calibration_key: str,
    ) -> CalibrationParameterBlock:
        for block in self.blocks:
            if block.calibration_key == calibration_key:
                return block

        raise KeyError(
            f'Unknown calibration key {calibration_key!r}.'
        )

    def parameter_index(
        self,
        calibration_key: str,
        parameter_name: str,
    ) -> int:
        '''Return the Fisher-matrix index for one calibration parameter.'''

        return self.block_for(
            calibration_key
        ).parameter_index(
            parameter_name
        )


@dataclass(frozen=True)
class NuisanceParameterBlock:
    '''Contiguous columns for one non-trajectory nuisance variable.

    Typical future examples are per-IMU gyroscope biases and, if needed,
    accelerometer biases.

    The representation is intentionally generic. Observability should not
    introduce another sensor registry or hard-code one particular nuisance
    model.
    '''

    nuisance_key: str
    parameter_names: tuple[str, ...]
    start: int

    def __post_init__(self) -> None:
        if not isinstance(self.nuisance_key, str) or not self.nuisance_key:
            raise ValueError(
                'nuisance_key must be a non-empty string.'
            )

        if self.start < 0:
            raise ValueError(
                'start must be nonnegative.'
            )

        if not self.parameter_names:
            raise ValueError(
                'A nuisance parameter block must contain at least '
                'one parameter.'
            )

        if any(
            not isinstance(parameter_name, str)
            or not parameter_name
            for parameter_name in self.parameter_names
        ):
            raise ValueError(
                'Nuisance parameter names must be non-empty strings.'
            )

        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise ValueError(
                'Nuisance parameter names must be unique within a block.'
            )

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    @property
    def stop(self) -> int:
        return self.start + self.dimension

    @property
    def parameter_slice(self) -> slice:
        return slice(
            self.start,
            self.stop,
        )

    def parameter_index(
        self,
        parameter_name: str,
    ) -> int:
        '''Return the nuisance-Jacobian column index of one parameter.'''

        try:
            local_index = self.parameter_names.index(parameter_name)
        except ValueError as exc:
            raise KeyError(
                f'Unknown nuisance parameter {parameter_name!r} '
                f'for key {self.nuisance_key!r}.'
            ) from exc

        return self.start + local_index


@dataclass(frozen=True)
class NuisanceParameterLayout:
    '''Ordering of non-trajectory variables marginalized from calibration.

    These variables participate in the sensor Jacobians but do not remain in
    the canonical calibration Fisher matrix.

    Trajectory-state columns are deliberately not included here because their
    number and ordering depend on the temporal window. Their layout belongs to
    the future linearization structures.
    '''

    blocks: tuple[NuisanceParameterBlock, ...] = ()

    def __post_init__(self) -> None:
        nuisance_keys = tuple(
            block.nuisance_key
            for block in self.blocks
        )

        if len(nuisance_keys) != len(set(nuisance_keys)):
            raise ValueError(
                'Nuisance layout keys must be unique.'
            )

        expected_start = 0

        for block in self.blocks:
            if block.start != expected_start:
                raise ValueError(
                    'Nuisance layout blocks must be contiguous '
                    'and start at zero.'
                )

            expected_start = block.stop

    @classmethod
    def empty(cls) -> 'NuisanceParameterLayout':
        '''Return an explicit layout with no nuisance variables.'''

        return cls()

    @classmethod
    def from_parameter_blocks(
        cls,
        parameter_blocks: Sequence[
            tuple[
                str,
                Sequence[str],
            ]
        ],
    ) -> 'NuisanceParameterLayout':
        '''Build a nuisance layout from ordered key/name specifications.

        Every item is

            (nuisance_key, parameter_names)

        For example, a future gyroscope-bias block can use a key associated
        with one particular IMU together with three explicitly ordered bias
        parameter names.
        '''

        blocks: list[NuisanceParameterBlock] = []
        start = 0

        for nuisance_key, parameter_names in parameter_blocks:
            parameter_names = tuple(parameter_names)

            block = NuisanceParameterBlock(
                nuisance_key=nuisance_key,
                parameter_names=parameter_names,
                start=start,
            )

            blocks.append(block)
            start = block.stop

        return cls(
            blocks=tuple(blocks)
        )

    @property
    def nuisance_keys(self) -> tuple[str, ...]:
        return tuple(
            block.nuisance_key
            for block in self.blocks
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        '''Return qualified nuisance names in Jacobian-column order.'''

        return tuple(
            f'{block.nuisance_key}.{parameter_name}'
            for block in self.blocks
            for parameter_name in block.parameter_names
        )

    @property
    def total_dimension(self) -> int:
        return sum(
            block.dimension
            for block in self.blocks
        )

    def block_for(
        self,
        nuisance_key: str,
    ) -> NuisanceParameterBlock:
        for block in self.blocks:
            if block.nuisance_key == nuisance_key:
                return block

        raise KeyError(
            f'Unknown nuisance key {nuisance_key!r}.'
        )

    def parameter_index(
        self,
        nuisance_key: str,
        parameter_name: str,
    ) -> int:
        '''Return the nuisance-Jacobian index for one parameter.'''

        return self.block_for(
            nuisance_key
        ).parameter_index(
            parameter_name
        )