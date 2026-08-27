"""Tests for the Table descriptor and its DDL generation."""

import duckdb
import polars as pl
import pytest

from fantasy_football.storage.table import Table, duckdb_type
from fantasy_football.storage.tables import MINUTES_PREDICTION

WIDGET = Table(
    name="widget",
    schema={
        "season": pl.Utf8,
        "element": pl.Int64,
        "label": pl.Utf8,
        "ratio": pl.Float64,
    },
    primary_key=("season", "element"),
    order_by=("season", "element"),
)


def test_columns_derive_from_schema_order():
    """``columns`` is the schema's keys in declaration order."""
    assert WIDGET.columns == ["season", "element", "label", "ratio"]


def test_ddl_marks_primary_key_columns_not_null():
    """Primary-key columns are NOT NULL; others are nullable."""
    assert "season VARCHAR NOT NULL" in WIDGET.ddl
    assert "element BIGINT NOT NULL" in WIDGET.ddl
    assert "label VARCHAR," in WIDGET.ddl
    assert "label VARCHAR NOT NULL" not in WIDGET.ddl


def test_ddl_declares_the_primary_key():
    """The generated DDL carries a composite PRIMARY KEY clause."""
    assert "PRIMARY KEY (season, element)" in WIDGET.ddl


def test_ddl_is_executable_and_builds_the_expected_table():
    """DuckDB accepts the generated DDL and builds the right columns."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        info = connection.execute("PRAGMA table_info('widget')").fetchall()
    finally:
        connection.close()
    assert [row[1] for row in info] == WIDGET.columns
    assert [row[2] for row in info] == [
        "VARCHAR",
        "BIGINT",
        "VARCHAR",
        "DOUBLE",
    ]


def test_ddl_is_idempotent_via_if_not_exists():
    """Executing the DDL twice does not raise."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        connection.execute(WIDGET.ddl)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (pl.Utf8, "VARCHAR"),
        (pl.Int64, "BIGINT"),
        (pl.Boolean, "BOOLEAN"),
        (pl.Float64, "DOUBLE"),
        (pl.Date, "DATE"),
        (pl.Datetime("us"), "TIMESTAMP"),
    ],
)
def test_duckdb_type_maps_every_dtype_in_use(dtype, expected):
    """Each Polars dtype the project stores maps to a DuckDB type."""
    assert duckdb_type(dtype) == expected


def test_duckdb_type_rejects_an_unmapped_dtype():
    """An unmapped dtype raises rather than emitting invalid SQL."""
    with pytest.raises(ValueError, match="No DuckDB type"):
        duckdb_type(pl.List(pl.Int64))


def test_coerce_selects_canonical_columns_and_drops_extras():
    """Coercion narrows to the schema and ignores unknown columns."""
    frame = pl.DataFrame(
        {
            "season": ["2024-25"],
            "element": [1],
            "label": ["a"],
            "ratio": [0.5],
            "junk": ["ignored"],
        }
    )
    assert WIDGET.coerce(frame).columns == WIDGET.columns


def test_coerce_pins_dtypes():
    """Coercion casts incoming dtypes to the declared schema."""
    frame = pl.DataFrame(
        {
            "season": ["2024-25"],
            "element": [1.0],
            "label": ["a"],
            "ratio": [1],
        }
    )
    shaped = WIDGET.coerce(frame)
    assert shaped.schema["element"] == pl.Int64
    assert shaped.schema["ratio"] == pl.Float64


def test_coerce_applies_the_normalise_hook_first():
    """A normalise hook runs before column selection."""
    table = Table(
        name="widget",
        schema={"season": pl.Utf8, "element": pl.Int64},
        primary_key=("season", "element"),
        order_by=("season",),
        normalise=lambda f: f.with_columns(pl.col("element") * 10),
    )
    frame = pl.DataFrame({"season": ["2024-25"], "element": [1]})
    assert table.coerce(frame)["element"].to_list() == [10]


def test_load_applies_the_enrich_hook():
    """An enrich hook runs after reading."""
    table = Table(
        name="widget",
        schema={"season": pl.Utf8, "element": pl.Int64},
        primary_key=("season", "element"),
        order_by=("season",),
        enrich=lambda f: f.with_columns(pl.col("element") + 100),
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(table.ddl)
        table.upsert_current(
            connection,
            pl.DataFrame({"season": ["2024-25"], "element": [1]}),
            "2024-25",
        )
        assert table.load(connection)["element"].to_list() == [101]
    finally:
        connection.close()


def test_load_skips_enrich_on_an_empty_table():
    """An empty read returns early without enriching."""
    table = Table(
        name="widget",
        schema={"season": pl.Utf8, "element": pl.Int64},
        primary_key=("season", "element"),
        order_by=("season",),
        enrich=lambda f: (_ for _ in ()).throw(AssertionError("enriched")),
    )
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(table.ddl)
        assert table.load(connection).is_empty()
    finally:
        connection.close()


def test_write_immutable_skips_a_season_already_present():
    """An immutable season is written once and never overwritten."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        first = pl.DataFrame(
            {
                "season": ["2024-25"],
                "element": [1],
                "label": ["first"],
                "ratio": [0.1],
            }
        )
        second = first.with_columns(pl.lit("second").alias("label"))
        WIDGET.write_immutable(connection, first, "2024-25")
        WIDGET.write_immutable(connection, second, "2024-25")
        assert WIDGET.load(connection)["label"].to_list() == ["first"]
    finally:
        connection.close()


def test_upsert_current_replaces_only_that_season():
    """Upserting a season rewrites it and leaves others intact."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        WIDGET.upsert_current(
            connection,
            pl.DataFrame(
                {
                    "season": ["2023-24", "2024-25"],
                    "element": [1, 1],
                    "label": ["old", "old"],
                    "ratio": [0.1, 0.1],
                }
            ),
            "2023-24",
        )
        WIDGET.upsert_current(
            connection,
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "element": [2],
                    "label": ["new"],
                    "ratio": [0.2],
                }
            ),
            "2024-25",
        )
        result = WIDGET.load(connection)
        assert sorted(result["label"].to_list()) == ["new", "old"]
    finally:
        connection.close()


def test_seasons_present_reports_stored_seasons():
    """``seasons_present`` returns the distinct seasons in the table."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        WIDGET.upsert_current(
            connection,
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "element": [1],
                    "label": ["a"],
                    "ratio": [0.1],
                }
            ),
            "2024-25",
        )
        assert WIDGET.seasons_present(connection) == {"2024-25"}
    finally:
        connection.close()


def test_replace_partition_scopes_delete_to_predicates(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Replacing one partition must leave sibling partitions intact."""
    existing = pl.DataFrame(
        {
            "season": ["2026-27"] * 2,
            "gw": [1, 1],
            "element": [1, 2],
            "opponent": [3, 3],
            "p_zero": [0.1, 0.2],
            "p_partial": [0.2, 0.2],
            "p_sixty_plus": [0.7, 0.6],
            "expected_minutes": [60.0, 55.0],
            "model_version": ["1", "1"],
            "prediction_kind": ["backfill", "forward"],
            "snapshot_captured_at": [None, None],
        }
    )
    MINUTES_PREDICTION.append(db, existing)

    replacement = existing.filter(
        pl.col("prediction_kind") == "forward"
    ).with_columns(expected_minutes=pl.lit(10.0))
    MINUTES_PREDICTION.replace_partition(
        db,
        replacement,
        equals={"season": "2026-27", "prediction_kind": "forward"},
    )

    stored = MINUTES_PREDICTION.load(db)
    by_kind = {
        row["prediction_kind"]: row["expected_minutes"]
        for row in stored.iter_rows(named=True)
    }
    assert by_kind["backfill"] == pytest.approx(60.0)
    assert by_kind["forward"] == pytest.approx(10.0)


def test_replace_partition_gw_from_preserves_earlier_gameweeks(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Frozen forward rows below the floor survive a rewrite."""
    existing = pl.DataFrame(
        {
            "season": ["2026-27"] * 2,
            "gw": [1, 2],
            "element": [1, 1],
            "opponent": [3, 4],
            "p_zero": [0.1, 0.1],
            "p_partial": [0.2, 0.2],
            "p_sixty_plus": [0.7, 0.7],
            "expected_minutes": [60.0, 60.0],
            "model_version": ["1", "1"],
            "prediction_kind": ["forward", "forward"],
            "snapshot_captured_at": [None, None],
        }
    )
    MINUTES_PREDICTION.append(db, existing)

    replacement = existing.filter(pl.col("gw") == 2).with_columns(
        expected_minutes=pl.lit(5.0)
    )
    MINUTES_PREDICTION.replace_partition(
        db,
        replacement,
        equals={"season": "2026-27", "prediction_kind": "forward"},
        gw_from=2,
    )

    stored = MINUTES_PREDICTION.load(db)
    by_gw = {
        row["gw"]: row["expected_minutes"]
        for row in stored.iter_rows(named=True)
    }
    # GW1 has kicked off; its forecast is frozen, not rewritten.
    assert by_gw[1] == pytest.approx(60.0)
    assert by_gw[2] == pytest.approx(5.0)


def test_conform_adds_missing_columns_as_typed_nulls():
    """A frame missing schema columns gains them as typed nulls."""
    frame = pl.DataFrame({"season": ["2016-17"], "element": [1]})
    conformed = WIDGET.conform(frame)
    assert set(conformed.columns) >= set(WIDGET.columns)
    assert conformed["label"].to_list() == [None]
    assert conformed["label"].dtype == pl.Utf8
    assert conformed["ratio"].dtype == pl.Float64


def test_conform_leaves_present_columns_untouched():
    """Columns already in the frame keep their values."""
    frame = pl.DataFrame(
        {
            "season": ["2016-17"],
            "element": [1],
            "label": ["keep me"],
            "ratio": [0.5],
        }
    )
    conformed = WIDGET.conform(frame)
    assert conformed["label"].to_list() == ["keep me"]
    assert conformed["ratio"].to_list() == [0.5]


def test_conform_keeps_extra_columns_for_coerce_to_drop():
    """Extra source columns survive conform; coerce is what narrows."""
    frame = pl.DataFrame({"season": ["2016-17"], "element": [1], "extra": [9]})
    assert "extra" in WIDGET.conform(frame).columns


def test_conform_output_is_coercible():
    """A conformed frame passes through coerce without raising."""
    frame = pl.DataFrame({"season": ["2016-17"], "element": [1], "extra": [9]})
    shaped = WIDGET.coerce(WIDGET.conform(frame))
    assert shaped.columns == WIDGET.columns
    assert shaped.height == 1


def test_conform_on_empty_frame_produces_empty_typed_frame():
    """An empty source frame conforms to an empty, fully-typed frame."""
    frame = pl.DataFrame({"season": [], "element": []})
    conformed = WIDGET.conform(frame)
    assert conformed.height == 0
    assert set(WIDGET.columns) <= set(conformed.columns)


def test_conform_on_a_zero_column_frame_produces_an_empty_typed_frame():
    """A frame with no columns at all conforms to height 0, not 1.

    ``pl.lit(None)`` broadcasts to a single row when there is no
    existing column to take the frame's height from, so a naive
    ``with_columns`` on a zero-column frame produces a phantom one-row
    frame instead of an empty one. The 0-row/2-column case above does
    not exercise this path.
    """
    frame = pl.DataFrame()
    conformed = WIDGET.conform(frame)
    assert conformed.height == 0
    assert set(WIDGET.columns) <= set(conformed.columns)


def test_unknown_columns_reports_undeclared_source_columns():
    """Columns absent from the schema are reported, sorted."""
    frame = pl.DataFrame(
        {"season": ["2016-17"], "element": [1], "zebra": [1], "apple": [2]}
    )
    assert WIDGET.unknown_columns(frame) == ["apple", "zebra"]


def test_unknown_columns_empty_when_frame_matches_schema():
    """A frame with only schema columns reports nothing unknown."""
    frame = pl.DataFrame(
        {"season": ["a"], "element": [1], "label": ["x"], "ratio": [1.0]}
    )
    assert WIDGET.unknown_columns(frame) == []


def test_replace_all_empties_the_table_first():
    """A wholesale rebuild leaves only the rows it was handed."""
    connection = duckdb.connect(":memory:")
    try:
        connection.execute(WIDGET.ddl)
        WIDGET.append(
            connection,
            pl.DataFrame(
                {
                    "season": ["2023-24"],
                    "element": [1],
                    "label": ["stale"],
                    "ratio": [0.1],
                }
            ),
        )
        WIDGET.replace_all(
            connection,
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "element": [2],
                    "label": ["fresh"],
                    "ratio": [0.2],
                }
            ),
        )
        stored = WIDGET.load(connection)
        assert stored["label"].to_list() == ["fresh"]
    finally:
        connection.close()


def test_ddl_declares_each_unique_group():
    """A unique group beyond the primary key reaches the create statement."""
    table = Table(
        name="widget_unique",
        schema={"season": pl.Utf8, "element": pl.Int64},
        primary_key=("season",),
        order_by=("season",),
        unique=(("element",),),
    )
    assert "UNIQUE (element)" in table.ddl
