import operator
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Union

import matplotlib
from schemdraw import Drawing, flow
import pandas as pd

from caumim.constants import (
    COLNAME_DELTA_INCLUSION_INTIME,
    COLNAME_DELTA_INTERVENTION_INCLUSION,
    COLNAME_DELTA_INTIME_ADMISSION,
    COLNAME_DELTA_MORTALITY,
    COLNAME_INCLUSION_START,
    COLNAME_INTERVENTION_START,
)

COMMON_DELTAS = [
    COLNAME_DELTA_MORTALITY,
    COLNAME_DELTA_INTERVENTION_INCLUSION,
    COLNAME_DELTA_INCLUSION_INTIME,
    COLNAME_DELTA_INTIME_ADMISSION,
    "los_hospital",
    "los_icu",
]


def add_delta(
    target_trial_population: pd.DataFrame,
):
    """Generate time deltas for target trial population in day unit.

    Args:
        target_trial_population (pd.DataFrame): _description_
        unit (str, optional): _description_. Defaults to "hours".
    Returns:
        _type_: _description_
    """
    unit_multiplier = 3600 * 24
    target_trial_population_w_delta = target_trial_population.copy()

    target_trial_population[COLNAME_DELTA_MORTALITY] = (
        target_trial_population["dod"]
        - target_trial_population[COLNAME_INCLUSION_START]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INTERVENTION_INCLUSION] = (
        target_trial_population[COLNAME_INTERVENTION_START]
        - target_trial_population[COLNAME_INCLUSION_START]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INCLUSION_INTIME] = (
        target_trial_population[COLNAME_INCLUSION_START]
        - target_trial_population["icu_intime"]
    ).dt.total_seconds() / unit_multiplier
    target_trial_population[COLNAME_DELTA_INTIME_ADMISSION] = (
        target_trial_population["icu_intime"]
        - target_trial_population["admittime"]
    ).dt.total_seconds() / unit_multiplier

    return target_trial_population_w_delta


def describe_delta(target_trial_population, unit="hours"):
    target_trial_population_ = target_trial_population.copy()
    if unit == "hours":
        for delta_ in COMMON_DELTAS:
            target_trial_population_[delta_] = (
                target_trial_population_[delta_] * 24
            )
    elif unit == "days":
        pass
    else:
        raise ValueError(f"Unit {unit} not supported.")
    return target_trial_population_[COMMON_DELTAS].describe().T


### flowchart stuff ###


def to_set(data: Iterable):
    if isinstance(data, pd.Series):
        ids = data.to_numpy()
    elif isinstance(data, pd.DataFrame):
        assert (
            len(data.columns) == 1
        ), "A DataFrame with more than 1 column was provided"
        ids = data[data.columns[0]].to_numpy()
    else:
        ids = data
    return set(ids)


class Data:
    def __init__(
        self,
        ids: Set[int],
    ):
        self.ids = ids
        self.n = len(ids)

    def __repr__(self) -> str:
        return f"n = {self.n}"

    def __str__(self) -> str:
        return self.__repr__()


class Criterion:
    def __init__(
        self,
        description: str,
        excluded_description: str,
        input_data: Data,
        output_data: Data,
        excluded_data: Data,
    ):
        self.description = description
        self.full_description = (
            description.strip() + "\n" + f"({output_data})"
        ).strip()
        self.excluded_description = (
            excluded_description.strip() + "\n" + f"({excluded_data})"
        ).strip()

        self.input_data = input_data
        self.output_data = output_data
        self.excluded_data = excluded_data

    def get_bbox(self, txt: Optional[str] = None, fontsize: int = 10):
        txt = txt if txt else self.full_description

        ffam = "dejavu sans"
        fp = matplotlib.font_manager.FontProperties(
            family=ffam,
            style="normal",
            size=fontsize,
            weight="normal",
            stretch="normal",
        )

        lines = txt.split("\n")
        bboxes = [
            matplotlib.textpath.TextPath(
                (100, 100), txt, prop=fp
            ).get_extents()
            for txt in lines
        ]

        max_w = max(bboxes, key=operator.attrgetter("width")).width
        max_h = max(bboxes, key=operator.attrgetter("height")).height

        SCALE_FACTOR = (
            fontsize / 10
        ) / 34.2  # 1/34.2 -> Computed for 'dejavu sans', size 10

        w = 0.5 + SCALE_FACTOR * max_w
        h = 0.5 + len(lines) * (SCALE_FACTOR * (max_h + 2))

        return dict(w=w, h=h)

    def draw(
        self,
        d: Drawing,
        **kwargs,
    ):
        arrow_length = kwargs.get("arrow_length", 3)
        direction = kwargs.get("direction", "right")
        final_split = kwargs.get("final_split", False)
        fontsize = kwargs.get("fontsize", 10)

        bbox_1 = self.get_bbox(fontsize=fontsize)
        bbox_2 = self.get_bbox(
            txt=self.excluded_description, fontsize=fontsize
        )

        if not final_split:
            condition = flow.Decision(w=0, h=0)
            d += flow.Line().down(1 / 2)
            d += condition
            d += getattr(flow.Arrow(), direction)(arrow_length).at(condition.E)

            d += (
                flow.Box(**bbox_2)
                .label(self.excluded_description)
                .linewidth(1)
            )
            d += flow.Arrow().at(condition.S).down(1 / 2)
            d += flow.Box(**bbox_1).label(self.full_description)

        else:
            arrow_length = 1 + max(bbox_1["w"], bbox_2["w"]) / 2

            d += flow.Line().down()
            intersection = flow.Box(w=0, h=0)
            left_intersection = flow.Box(w=0, h=0)
            right_intersection = flow.Box(w=0, h=0)

            d += intersection

            d += flow.Line().at(intersection.W).left(arrow_length)
            d += left_intersection.anchor("E")
            d += flow.Arrow().down().at(left_intersection.S)
            d += (
                flow.Box(**bbox_1)
                .label(self.full_description)
                .label(self.left_title, loc="bottom", ofst=0.2)
            )

            d += flow.Line().at(intersection.E).right(arrow_length)
            d += right_intersection.anchor("W")
            d += flow.Arrow().down().at(right_intersection.S)
            d += (
                flow.Box(**bbox_2)
                .label(self.excluded_description)
                .label(self.right_title, loc="bottom", ofst=0.2)
            )

        return d

    def __repr__(self) -> str:
        return self.description

    def __str__(self) -> str:
        return self.__repr__()


class Flowchart:
    def __init__(
        self,
        initial_description: str,
        data: Union[pd.DataFrame, Dict[str, Iterable]],
        concat_criterion_description: bool = True,
        to_count: str = "person_id",
        cohort_characteristics: pd.DataFrame = None,
    ):
        """
        Main class to define an flowchart (inclusion diagram), extracted from
        [eds-scikit](https://github.com/aphp/eds-scikit) to avoid having to
        install all dependencies.

        Parameters
        ----------
        initial_description : str
            Description of the initial population
        data : Union[DataFrame, Dict[str, Iterable]]
            Either a Pandas/Koalas DataFrame, or a dictionary of iterables. If a
            dictionary, the initial cohort should be proivided under the
            **initial** key.
        concat_criterion_description : bool, optional
            Whether to concatenate provided description together when adding
            multiple criteria
        to_count : str, optional
            Only if `data` is a DataFrame: column of `data` from which the count
            is computed. Usually, this will be the column containing patient or
            stay IDs.
        cohort_characteristics : DataFrame, optional
            If provided, the columns will be used to summarize excluded patients.
        """

        self.initial_description = initial_description
        self.data = data
        self.to_count = to_count

        self.check_data()

        self.ids = self.get_unique()
        self.criteria = []
        self.concat_criterion_description = concat_criterion_description

        self.final_split = None

        self.drawing = None
        if cohort_characteristics is not None:
            if self.to_count not in cohort_characteristics.columns:
                raise ValueError(
                    f"{to_count} is not a column of `cohort characteristics`"
                )
            self.cohort_characteristics = cohort_characteristics

    def check_data(self):
        if isinstance(self.data, pd.DataFrame):
            if self.to_count not in self.data.columns:
                raise ValueError(
                    f"The parameter `to_count` ({self.to_count}) "
                    "is not a column of `data`"
                )
        elif isinstance(self.data, dict):
            if "initial" not in self.data.keys():
                raise ValueError(
                    "With `data` provided as a dictionary, "
                    "the initial cohort should be provided ",
                    'under the `"initial"` key: '
                    "`data['initial'] = my_initial_cohort",
                )
        else:
            raise TypeError(
                "The provided `data` should be a DataFrame or "
                f"a dictionary, not a {type(self.data)}"
            )

    def get_unique(self, criterion_name: Optional[str] = None):
        if isinstance(self.data, pd.DataFrame):
            ids = (
                self.data[self.to_count]
                if criterion_name is None
                else self.data[self.data[criterion_name]][self.to_count]
            )

        else:
            ids = (
                self.data["initial"]
                if criterion_name is None
                else self.data[criterion_name]
            )

        return to_set(ids)

    def get_last_description(self) -> str:
        return (
            "" if not self.criteria else (self.criteria[-1].description + "\n")
        )

    def add_criterion(
        self,
        description: str,
        criterion_name: str,
        excluded_description: str = "",
    ):
        """
        Adds a criterion to the flowchart

        ![](../../../_static/flowchart/criterion.png)

        Parameters
        ----------
        description : str
            Description of the cohort passing the criterion
        criterion_name : str

            - If `data` is a DataFrame, `criterion_name` is a
            boolean column of `data` to split between
            passing cohort (`data[criterion_name] == True`) and
            excluded column (`data[criterion_name] == False`)
            - If `data` is a dictionary, `criterion_name` is a
            key of `data` containing the passing cohort as an iterable
            of IDs (list, set , Series, array, etc.)
        excluded_description: str
            Description of the cohort excluded by the criterion
        """

        input_data = (
            Data(
                self.ids,
            )
            if not self.criteria
            else self.criteria[-1].output_data
        )

        passing_criterion_ids = self.get_unique(criterion_name=criterion_name)

        output_data = Data(
            passing_criterion_ids & input_data.ids,
        )
        excluded_data = Data(
            input_data.ids - passing_criterion_ids,
        )

        description = (
            description
            if not self.concat_criterion_description
            else (self.get_last_description() + description)
        )
        if self.cohort_characteristics is not None:
            excluded_description = self.get_summary_characteristics(
                excluded_data.ids
            )
        added_criterion = Criterion(
            description=description,
            excluded_description=excluded_description,
            input_data=input_data,
            output_data=output_data,
            excluded_data=excluded_data,
        )
        self.criteria.append(added_criterion)

    def add_final_split(
        self,
        left_description: str,
        right_description: str,
        criterion_name: str,
        left_title: str = "",
        right_title: str = "",
    ):
        """
        Adds a final split in two distinct cohorts.
        Should be called after all other critera were added.

        ![](../../../_static/flowchart/split.png)

        Parameters
        ----------
        left_description : str
            Description of the left cohort
        right_description : str
            Description of the right cohort
        criterion_name : str

            - If `data` is a DataFrame, `criterion_name` is a
            boolean column of `data` to split between
            passing cohort (`data[criterion_name] == True`) and
            excluded column (`data[criterion_name] == False`)
            - If `data` is a dictionary, `criterion_name` is a
            key of `data` containing the passing cohort as an iterable
            of IDs (list, set , Series, array, etc.)
        left_title : str, optional
            Title of the left cohort
        right_title : str, optional
            title of the right cohort
        """
        input_data = (
            Data(
                self.ids,
            )
            if not self.criteria
            else self.criteria[-1].output_data
        )

        left_criterion_ids = self.get_unique(criterion_name=criterion_name)

        left_data = Data(
            left_criterion_ids & input_data.ids,
        )
        right_data = Data(
            input_data.ids - left_criterion_ids,
        )

        left_description = (
            left_description
            if not self.concat_criterion_description
            else (self.get_last_description() + left_description)
        )

        right_description = (
            right_description
            if not self.concat_criterion_description
            else (self.get_last_description() + right_description)
        )

        added_criterion = Criterion(
            description=left_description,
            excluded_description=right_description,
            input_data=input_data,
            output_data=left_data,
            excluded_data=right_data,
        )
        added_criterion.left_title = left_title
        added_criterion.right_title = right_title

        self.final_split = added_criterion

    def generate_flowchart(
        self,
        alternate: bool = False,
        fontsize: int = 10,
    ):
        """
        Generate and display the flowchart

        Parameters
        ----------
        alternate : bool, optional
            Wether to alternate the excluded box positions
        fontsize : int, optional
            fontsize
        """
        max_criterion_width = max(
            [c.get_bbox(fontsize=fontsize)["w"] for c in self.criteria]
        )

        arrow_length = 1.2 * (max_criterion_width / 2)

        directions = ["right", "left"] if alternate else ["right", "right"]

        d = Drawing()
        d.config(font="dejavu sans", fontsize=fontsize, unit=1)

        start_description = (
            self.initial_description
            + "\n"
            + f"({self.criteria[0].input_data})"
        )
        start_bbox = Criterion.get_bbox(None, txt=start_description)

        d += flow.Start(**start_bbox).label(start_description)
        for i, c in enumerate(self.criteria):
            d = c.draw(
                d,
                arrow_length=arrow_length,
                direction=directions[i % 2],
                fontsize=fontsize,
            )
        if self.final_split is not None:
            d = self.final_split.draw(d, final_split=True, fontsize=fontsize)

        self.drawing = d

        return d

    def get_summary_characteristics(self, patient_ids: Set[str]) -> str:
        """
        Compute summary characteristics of a subsest of the patient IDs.
        Only support binary characteristics, eg. Male/Female.
        """
        patient_characteristics = (
            self.cohort_characteristics.merge(
                pd.DataFrame({self.to_count: list(patient_ids)}),
                on=self.to_count,
                how="inner",
            )
            .drop(columns=self.to_count)
            .mean()
        )
        summary_str = ""
        for col in patient_characteristics.index:
            summary_str += f"{col}: {patient_characteristics[col]:.2f}\n"
        return summary_str

    def save(
        self,
        filename: Union[str, Path],
        transparent: bool = False,
        dpi: int = 72,
    ):
        """
        Save the generated flowchart

        Parameters
        ----------
        filename : Union[str, Path]
            path to the saved file (should end with svg or png)
        transparent : bool, optional
            Wether to use a transparent background or not
        dpi : int, optional
            Resolution (only when saving png)
        """
        self.drawing.save(fname=filename, transparent=transparent, dpi=dpi)
