from __future__ import annotations

import math

import folium
import pandas as pd
from folium.plugins import FastMarkerCluster, HeatMap, MiniMap

from src.paths import PROJECT_ROOT, ZONE_RISK_PATH
from src.zones import DEFAULT_GRID_SIZE


OUTPUT_MAP_PATH = PROJECT_ROOT / "zone_risk_map.html"
LOW_RISK_COLOR = "#2e8b57"
MODERATE_RISK_COLOR = "#f59e0b"
HIGH_RISK_COLOR = "#dc2626"
BOUNDARY_LIMIT_PER_GRADE = 700


def _score_column(zone_risk: pd.DataFrame) -> str:
    return "combined_risk_score" if "combined_risk_score" in zone_risk.columns else "risk_score"


def _classify_grade(score: float, low_cutoff: float, moderate_cutoff: float) -> tuple[str, str]:
    if score < low_cutoff:
        return "Low", LOW_RISK_COLOR
    if score < moderate_cutoff:
        return "Moderate", MODERATE_RISK_COLOR
    return "High", HIGH_RISK_COLOR


def _build_popup(row: pd.Series, score_column: str, grade: str) -> str:
    return (
        f"<div style='font-size: 13px;'>"
        f"<strong>{grade} Risk Zone</strong><br>"
        f"Score: {row[score_column]:.2f}<br>"
        f"Historical Risk Score: {row['risk_score']}<br>"
        f"Accidents: {int(row['accident_count'])}<br>"
        f"Fatal: {int(row['fatal_count'])}<br>"
        f"Serious: {int(row['serious_count'])}<br>"
        f"Slight: {int(row['slight_count'])}<br>"
        f"Pred Fatal Risk: {row.get('predicted_fatal_risk', 0):.3f}<br>"
        f"Pred Severe Risk: {row.get('predicted_severe_risk', 0):.3f}"
        f"</div>"
    )


def _add_legend(danger_map: folium.Map, low_cutoff: float, moderate_cutoff: float) -> None:
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 28px;
        left: 28px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.12);
        font-family: Arial, sans-serif;
        font-size: 13px;
        color: #111827;
        min-width: 238px;
    ">
        <div style="font-weight: 700; margin-bottom: 8px;">Risk Grade</div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display:inline-block; width: 12px; height: 12px; background: {LOW_RISK_COLOR}; border-radius: 50%; margin-right: 8px;"></span>
            Low prone area: score &lt; {low_cutoff:.1f}
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display:inline-block; width: 12px; height: 12px; background: {MODERATE_RISK_COLOR}; border-radius: 50%; margin-right: 8px;"></span>
            Moderate prone area: {low_cutoff:.1f} to {moderate_cutoff:.1f}
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="display:inline-block; width: 12px; height: 12px; background: {HIGH_RISK_COLOR}; border-radius: 50%; margin-right: 8px;"></span>
            High prone area: score &gt; {moderate_cutoff:.1f}
        </div>
        <div style="font-size: 11px; color: #4b5563; margin-top: 8px;">
            Heatmap shows all zones. Boundary layers highlight sample zones per grade.
        </div>
    </div>
    """
    danger_map.get_root().html.add_child(folium.Element(legend_html))


def _add_title_card(danger_map: folium.Map, zone_count: int, low_cutoff: float, moderate_cutoff: float) -> None:
    title_html = f"""
    <div style="
        position: fixed;
        top: 18px;
        left: 22px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        font-family: Arial, sans-serif;
        color: #111827;
        width: 330px;
    ">
        <div style="font-size: 18px; font-weight: 700; margin-bottom: 6px;">Accident Risk Explorer</div>
        <div style="font-size: 13px; line-height: 1.45; color: #4b5563;">
            Green shows low prone areas, orange shows moderate prone areas, and red shows high prone areas.
            Use the layer control to toggle heatmap, markers, and zone boundaries.
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #374151;">
            Zones plotted: <strong>{zone_count:,}</strong><br>
            Score cutoffs: <strong>{low_cutoff:.1f}</strong> and <strong>{moderate_cutoff:.1f}</strong>
        </div>
    </div>
    """
    danger_map.get_root().html.add_child(folium.Element(title_html))


def _add_heatmap_layer(danger_map: folium.Map, zone_risk: pd.DataFrame) -> None:
    heat_data = zone_risk[["avg_lat", "avg_lon", "accident_count"]].values.tolist()
    HeatMap(
        heat_data,
        name="Accident intensity heatmap",
        min_opacity=0.28,
        radius=18,
        blur=20,
        gradient={
            0.20: LOW_RISK_COLOR,
            0.55: MODERATE_RISK_COLOR,
            0.85: HIGH_RISK_COLOR,
        },
        show=True,
    ).add_to(danger_map)


def _add_marker_cluster(danger_map: folium.Map, marker_rows: list[list[object]]) -> None:
    callback = """
    function (row) {
        var marker = L.circleMarker([row[0], row[1]], {
            color: row[2],
            fillColor: row[2],
            fillOpacity: 0.72,
            opacity: 0.9,
            radius: row[3],
            weight: 1
        });
        marker.bindTooltip(row[4] + " risk | Score: " + row[5], {sticky: false});
        marker.bindPopup(row[6], {maxWidth: 280});
        return marker;
    }
    """
    FastMarkerCluster(marker_rows, callback=callback, name="All risk markers").add_to(danger_map)


def _boundary_feature(row: pd.Series, score_column: str, grade: str, color: str) -> dict:
    lat_center = float(row["lat_grid"]) * DEFAULT_GRID_SIZE
    lon_center = float(row["lon_grid"]) * DEFAULT_GRID_SIZE
    half_step = DEFAULT_GRID_SIZE / 2
    popup = _build_popup(row, score_column, grade)
    return {
        "type": "Feature",
        "properties": {
            "popup": popup,
            "tooltip": f"{grade} | Score {row[score_column]:.2f}",
            "style": {
                "fillColor": color,
                "color": color,
                "fillOpacity": 0.28,
                "opacity": 0.9,
                "weight": 1.0,
            },
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon_center - half_step, lat_center - half_step],
                [lon_center + half_step, lat_center - half_step],
                [lon_center + half_step, lat_center + half_step],
                [lon_center - half_step, lat_center + half_step],
                [lon_center - half_step, lat_center - half_step],
            ]],
        },
    }


def _add_boundary_layers(danger_map: folium.Map, zone_risk: pd.DataFrame, score_column: str) -> None:
    grade_specs = [
        ("Low", LOW_RISK_COLOR, "Low risk boundaries"),
        ("Moderate", MODERATE_RISK_COLOR, "Moderate risk boundaries"),
        ("High", HIGH_RISK_COLOR, "High risk boundaries"),
    ]

    for grade, color, layer_name in grade_specs:
        grade_rows = zone_risk[zone_risk["risk_grade"] == grade].nlargest(BOUNDARY_LIMIT_PER_GRADE, score_column)
        features = [_boundary_feature(row, score_column, grade, color) for _, row in grade_rows.iterrows()]
        if not features:
            continue
        feature_group = folium.FeatureGroup(name=layer_name, show=(grade != "Low"))
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function=lambda feature: feature["properties"]["style"],
            tooltip=folium.GeoJsonTooltip(fields=["tooltip"], aliases=["Zone"], sticky=False),
            popup=folium.GeoJsonPopup(fields=["popup"], labels=False),
            highlight_function=lambda feature: {
                "weight": 2,
                "fillOpacity": 0.42,
                "color": feature["properties"]["style"]["color"],
            },
        ).add_to(feature_group)
        feature_group.add_to(danger_map)


def main() -> None:
    zone_risk = pd.read_csv(ZONE_RISK_PATH).dropna(subset=["avg_lat", "avg_lon", "lat_grid", "lon_grid"])
    score_column = _score_column(zone_risk)
    low_cutoff = float(zone_risk[score_column].quantile(0.33))
    moderate_cutoff = float(zone_risk[score_column].quantile(0.66))

    grades = zone_risk[score_column].apply(lambda score: _classify_grade(float(score), low_cutoff, moderate_cutoff))
    zone_risk["risk_grade"] = grades.str[0]
    zone_risk["risk_color"] = grades.str[1]

    center_lat = zone_risk["avg_lat"].mean()
    center_lon = zone_risk["avg_lon"].mean()
    danger_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles=None)

    light_tiles = folium.TileLayer("CartoDB positron", name="Light basemap", show=True)
    dark_tiles = folium.TileLayer("CartoDB dark_matter", name="Dark basemap", show=False)
    light_tiles.add_to(danger_map)
    dark_tiles.add_to(danger_map)

    max_accidents = max(int(zone_risk["accident_count"].max()), 1)
    marker_rows: list[list[object]] = []
    for _, row in zone_risk.iterrows():
        radius = 3 + (4 * math.sqrt(int(row["accident_count"]) / max_accidents))
        popup = _build_popup(row, score_column, str(row["risk_grade"]))
        marker_rows.append(
            [
                float(row["avg_lat"]),
                float(row["avg_lon"]),
                str(row["risk_color"]),
                round(radius, 2),
                str(row["risk_grade"]),
                round(float(row[score_column]), 2),
                popup,
            ]
        )

    _add_heatmap_layer(danger_map, zone_risk)
    _add_marker_cluster(danger_map, marker_rows)
    _add_boundary_layers(danger_map, zone_risk, score_column)
    _add_title_card(danger_map, len(zone_risk), low_cutoff, moderate_cutoff)
    _add_legend(danger_map, low_cutoff, moderate_cutoff)
    MiniMap(tile_layer=light_tiles, toggle_display=True, position="bottomright").add_to(danger_map)
    folium.LayerControl(collapsed=False).add_to(danger_map)

    danger_map.save(OUTPUT_MAP_PATH)
    print(f"Saved interactive map to {OUTPUT_MAP_PATH}")


if __name__ == "__main__":
    main()
