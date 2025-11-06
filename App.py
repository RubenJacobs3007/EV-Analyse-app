# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, FastMarkerCluster, Fullscreen
from shapely.geometry import Point
import geopandas as gpd
from streamlit_folium import st_folium
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy.optimize import curve_fit
import os

st.set_page_config(page_title="Laadpalen, Laadsessies & Voertuigdata Nederland", layout="wide")

# -------------------------
# Sidebar navigatie
# -------------------------
st.sidebar.title("🔀 Navigatie")
pagina = st.sidebar.radio(
    "Kies een onderdeel:",
    ["Voorpagina", "🗺️ Kaart van laadpalen", "️🗺️👴 Kaart van laapalen (oud)", "📊 Analyse van laadsessies", "🚗 Analyse van voertuigdata", "🚗👴 Analyse van voertuigdata (oud)"]
)

# -------------------------
# 🌐 Laadpalen data ophalen (voor alle pagina's)
# -------------------------
@st.cache_data(ttl=86400, show_spinner="Data ophalen van OpenChargeMap...")
def laad_data():
    aantal_punten = 10000
    api_key = "93b912b5-9d70-4b1f-960b-fb80a4c9c017"
    url_poi = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults={aantal_punten}&compact=true&verbose=false&key={api_key}"
    url_ref = f"https://api.openchargemap.io/v3/referencedata/?output=json&key={api_key}"

    response = requests.get(url_poi, timeout=60)
    response.raise_for_status()
    reference = requests.get(url_ref, timeout=60)
    reference.raise_for_status()

    responsejson = response.json()
    Laadpalen = pd.json_normalize(responsejson)
    ref_data = reference.json()

    connection_types = {c["ID"]: c["Title"] for c in ref_data.get("ConnectionTypes", [])}
    current_types = {c["ID"]: c["Title"] for c in ref_data.get("CurrentTypes", [])}

    Laadpalen = Laadpalen[
        Laadpalen["AddressInfo.Latitude"].notnull() & Laadpalen["AddressInfo.Longitude"].notnull()
    ].reset_index(drop=True)

    try:
        df4 = pd.json_normalize(Laadpalen["Connections"])
        df5 = pd.json_normalize(df4[0])
        Laadpalen = pd.concat([Laadpalen.reset_index(drop=True), df5.reset_index(drop=True)], axis=1)
    except Exception:
        pass

    mask = Laadpalen["Connections"].apply(lambda x: isinstance(x, list) and len(x) == 0)
    Laadpalen.drop(index=Laadpalen[mask].index, inplace=True)

    provincies_gdf = gpd.read_file("https://cartomap.github.io/nl/wgs84/provincie_2022.geojson").to_crs(epsg=4326)
    geometry = [Point(xy) for xy in zip(Laadpalen["AddressInfo.Longitude"], Laadpalen["AddressInfo.Latitude"])]
    laadpalen_gdf = gpd.GeoDataFrame(Laadpalen, geometry=geometry, crs="EPSG:4326")

    try:
        laadpalen_met_prov = gpd.sjoin(
            laadpalen_gdf, provincies_gdf[["statnaam", "geometry"]],
            how="left", predicate="within"
        )
    except TypeError:
        laadpalen_met_prov = gpd.sjoin(
            laadpalen_gdf, provincies_gdf[["statnaam", "geometry"]],
            how="left", op="within"
        )

    return laadpalen_met_prov, provincies_gdf, connection_types, current_types

# Data laden (eenmalig, beschikbaar voor alle pagina’s)
laadpalen_met_prov, provincies_gdf, connection_types, current_types = laad_data()
# -------------------------
# Voorpagina
# -------------------------
if pagina == "Voorpagina":
    st.markdown("""
    # Veranderingen op de Laadpalen en Elektrisch vervoer case
    > Team 13, Ruben en Jarno

    Welkom op dit dashboard. Hier tonen we de oude en de nieuwe versie van onze case over laadpalen en elektrisch vervoer. In de nieuwe versie hebben we meer richting en samenhang aangebracht dan in de eerste.

    ## Verbeteringen
    We hebben de case meer samenhang gegeven. Het doel is nu om te laten zien hoe het aantal elektrische auto’s de afgelopen jaren is gegroeid.
    ### Kaart
    Hier is een jaar selector toegevoegd bij de kaart en er is een grafiek die laat zien hoeveel laatpalen elk jaar zijn toegevoegd.
    ### Analyse van voertuigdata
    Hier gebruiken we in plaats van het bestand cars.pkl een API met dezelfde gegevens, maar nu over de periode 2010 tot en met 2025. We tonen bovendien de groei van het aantal elektrische auto’s per jaar met een Gompertz-fit, in plaats van een scatterplot van het aantal merken per jaar met een lineaire regressie.
    """)
# -------------------------
# 🗺️ Kaart van laadpalen
# -------------------------
if pagina == "🗺️ Kaart van laadpalen":
    st.title("🗺️ Laadpalenkaart Nederland")
    st.markdown("Selecteer provincies via dropdown menu of layer control.")

    # Provinciekeuze
    alle_provincies = sorted(provincies_gdf["statnaam"].unique())
    opties = ["Alle provincies"] + alle_provincies
    gekozen_provincie = st.selectbox("Kies provincie (één) of 'Alle provincies':", opties, index=0)

    # Kaart genereren
    default_center = [52.1326, 5.2913]
    m = folium.Map(location=default_center, zoom_start=8, min_zoom=7)
    Fullscreen().add_to(m)

    folium.GeoJson(
        provincies_gdf.to_json(),
        name="Provinciegrenzen",
        style_function=lambda x: {"fillColor": "#00000000", "color": "#555555", "weight": 1.2},
    ).add_to(m)

    # Fit bounds
    try:
        if gekozen_provincie == "Alle provincies":
            minx, miny, maxx, maxy = provincies_gdf.total_bounds
        else:
            geom = provincies_gdf.loc[provincies_gdf["statnaam"] == gekozen_provincie, "geometry"].values[0]
            minx, miny, maxx, maxy = geom.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception:
        pass
    # -----------------------------
    # 📅 Tijdslider voor laadpalen
    # -----------------------------
    if "DateCreated" in laadpalen_met_prov.columns:
        # Zet DateCreated om naar datetime (indien nog niet)
        laadpalen_met_prov["DateCreated"] = pd.to_datetime(laadpalen_met_prov["DateCreated"], errors="coerce")

        # Filter op geldige datums
        laadpalen_met_prov = laadpalen_met_prov.dropna(subset=["DateCreated"])

        # Definieer bereik van jaren
        min_year = int(laadpalen_met_prov["DateCreated"].dt.year.min())
        max_year = int(laadpalen_met_prov["DateCreated"].dt.year.max())


        geselecteerd_jaar = st.slider(
            "📆 Toon laadpalen tot en met jaar:",
            min_value=2011,
            max_value=2025,
            value=max_year,
            step=1,
            help="Gebruik de slider om te zien hoeveel laadpalen er in een bepaald jaar al bestonden."
        )

        # Filter laadpalen tot en met het geselecteerde jaar
        laadpalen_met_prov = laadpalen_met_prov[
            laadpalen_met_prov["DateCreated"].dt.year <= geselecteerd_jaar
            ]
    else:
        st.warning("Kolom 'DateCreated' ontbreekt in de data, tijdfilter wordt overgeslagen.")
    # Markers
    to_show = alle_provincies if gekozen_provincie == "Alle provincies" else [gekozen_provincie]
    for prov in to_show:
        subset = laadpalen_met_prov[laadpalen_met_prov["statnaam"] == prov]
        if subset.empty:
            continue
        cluster = MarkerCluster(name=prov, options={"maxClusterRadius": 200}).add_to(m)

        for _, row in subset.iterrows():
            lat, lon = row.get("AddressInfo.Latitude"), row.get("AddressInfo.Longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            connections = row["Connections"] if isinstance(row.get("Connections"), list) else []
            n_conn = len(connections)
            color = "green" if n_conn == 1 else "blue" if n_conn <= 3 else "red"
            popup_html = f"<h4>{row.get('AddressInfo.Title', 'Laadpaal')}</h4>"
            if connections:
                popup_html += "<h5>Aansluitingen:</h5><table style='width:100%;border-collapse:collapse;'>"
                popup_html += "<tr><th>Type</th><th>Vermogen (kW)</th><th>Stroom</th></tr>"
                for conn in connections:
                    conn_type = connection_types.get(conn.get("ConnectionTypeID"), "Onbekend")
                    power = conn.get("PowerKW", "Onbekend")
                    current_type = current_types.get(conn.get("CurrentTypeID"), "Onbekend")
                    popup_html += f"<tr><td>{conn_type}</td><td>{power}</td><td>{current_type}</td></tr>"
                popup_html += "</table>"
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=420),
                icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
                tooltip=row.get("AddressInfo.Title", ""),
            ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
        background-color: white; border:2px solid grey; z-index:9999;
        font-size:14px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <b>Legenda:</b><br>
    <i class="fa fa-map-marker fa-2x" style="color:green"></i> 1 aansluiting<br>
    <i class="fa fa-map-marker fa-2x" style="color:blue"></i> 2–3 aansluitingen<br>
    <i class="fa fa-map-marker fa-2x" style="color:red"></i> ≥4 aansluitingen
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=1300, height=800, returned_objects=[])

    if "DateCreated" in laadpalen_met_prov.columns:
        laadpalen_met_prov["DateCreated"] = pd.to_datetime(laadpalen_met_prov["DateCreated"], errors="coerce")
        aantal_per_jaar = laadpalen_met_prov["DateCreated"].dt.year.value_counts().sort_index()
        fig_laadsessies = go.Figure()
        fig_laadsessies.add_trace(go.Scatter(
            x=aantal_per_jaar.index,
            y=aantal_per_jaar.values,
            mode='lines+markers',
            name='Aantal laadpalen per jaar'
        ))
        fig_laadsessies.update_layout(
            title="Aantal laadpalen per jaar",
            xaxis_title="Jaar",
            yaxis_title="Aantal laadpalen",
            template="plotly_white"
        )
        st.plotly_chart(fig_laadsessies, use_container_width=True)
# -------------------------
# 🗺️ Kaart van laadpalen (oud)
# -------------------------
if pagina == "🗺️👴 Kaart van laadpalen (oud)":
    st.title("🗺️👴 Laadpalenkaart Nederland (oude versie)")
    st.markdown(
        "Selecteer provincies via dropdown menu of layer control.")


    @st.cache_data(ttl=86400, show_spinner="Data ophalen van OpenChargeMap...")
    def laad_data():
        aantal_punten = 10000
        # Let op: gebruik van een hardcoded API-sleutel in een publieke app is niet aanbevolen.
        api_key = "93b912b5-9d70-4b1f-960b-fb80a4c9c017"
        url_poi = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults={aantal_punten}&compact=true&verbose=false&key={api_key}"
        url_ref = f"https://api.openchargemap.io/v3/referencedata/?output=json&key={api_key}"

        response = requests.get(url_poi, timeout=60)
        response.raise_for_status()
        reference = requests.get(url_ref, timeout=60)
        reference.raise_for_status()

        responsejson = response.json()
        Laadpalen = pd.json_normalize(responsejson)
        ref_data = reference.json()

        connection_types = {c["ID"]: c["Title"] for c in ref_data.get("ConnectionTypes", [])}
        current_types = {c["ID"]: c["Title"] for c in ref_data.get("CurrentTypes", [])}

        Laadpalen = Laadpalen[
            Laadpalen["AddressInfo.Latitude"].notnull() &
            Laadpalen["AddressInfo.Longitude"].notnull()
            ].reset_index(drop=True)

        try:
            df4 = pd.json_normalize(Laadpalen["Connections"])
            df5 = pd.json_normalize(df4[0])
            Laadpalen = pd.concat([Laadpalen.reset_index(drop=True), df5.reset_index(drop=True)], axis=1)
        except Exception:
            pass

        mask = Laadpalen["Connections"].apply(lambda x: isinstance(x, list) and len(x) == 0)
        Laadpalen.drop(index=Laadpalen[mask].index, inplace=True)

        # Let op: de GeoJSON URL moet bereikbaar zijn of het bestand lokaal aanwezig zijn
        provincies_gdf = gpd.read_file("https://cartomap.github.io/nl/wgs84/provincie_2022.geojson").to_crs(epsg=4326)
        geometry = [Point(xy) for xy in zip(Laadpalen["AddressInfo.Longitude"], Laadpalen["AddressInfo.Latitude"])]
        laadpalen_gdf = gpd.GeoDataFrame(Laadpalen, geometry=geometry, crs="EPSG:4326")

        try:
            laadpalen_met_prov = gpd.sjoin(
                laadpalen_gdf, provincies_gdf[["statnaam", "geometry"]],
                how="left", predicate="within"
            )
        except TypeError:
            laadpalen_met_prov = gpd.sjoin(
                laadpalen_gdf, provincies_gdf[["statnaam", "geometry"]],
                how="left", op="within"
            )

        return laadpalen_met_prov, provincies_gdf, connection_types, current_types


    laadpalen_met_prov, provincies_gdf, connection_types, current_types = laad_data()

    # Provinciekeuze
    alle_provincies = sorted(provincies_gdf["statnaam"].unique())
    opties = ["Alle provincies"] + alle_provincies
    gekozen_provincie = st.selectbox("Kies provincie (één) of 'Alle provincies':", opties, index=0)

    # Kaart genereren
    default_center = [52.1326, 5.2913]
    m = folium.Map(location=default_center, zoom_start=8, min_zoom=7)
    Fullscreen().add_to(m)

    folium.GeoJson(
        provincies_gdf.to_json(),
        name="Provinciegrenzen",
        style_function=lambda x: {"fillColor": "#00000000", "color": "#555555", "weight": 1.2},
    ).add_to(m)

    try:
        if gekozen_provincie == "Alle provincies":
            minx, miny, maxx, maxy = provincies_gdf.total_bounds
        else:
            geom = provincies_gdf.loc[provincies_gdf["statnaam"] == gekozen_provincie, "geometry"].values[0]
            minx, miny, maxx, maxy = geom.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
    except Exception:
        pass

    # Markers
    to_show = alle_provincies if gekozen_provincie == "Alle provincies" else [gekozen_provincie]
    for prov in to_show:
        subset = laadpalen_met_prov[laadpalen_met_prov["statnaam"] == prov]
        if subset.empty:
            continue

        cluster = MarkerCluster(name=prov, options={"maxClusterRadius": 200}).add_to(m)

        for _, row in subset.iterrows():
            lat, lon = row.get("AddressInfo.Latitude"), row.get("AddressInfo.Longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue

            connections = row["Connections"] if isinstance(row.get("Connections"), list) else []
            n_conn = len(connections)
            if n_conn == 1:
                color = "green"
            elif n_conn <= 3:
                color = "blue"
            else:
                color = "red"

            popup_html = f"<h4>{row.get('AddressInfo.Title', 'Laadpaal')}</h4>"
            if connections:
                popup_html += "<h5>Aansluitingen:</h5><table style='width:100%;border-collapse:collapse;'>"
                popup_html += "<tr><th>Type</th><th>Vermogen (kW)</th><th>Stroom</th></tr>"
                for conn in connections:
                    conn_type = connection_types.get(conn.get("ConnectionTypeID"), "Onbekend")
                    power = conn.get("PowerKW", "Onbekend")
                    current_type = current_types.get(conn.get("CurrentTypeID"), "Onbekend")
                    popup_html += f"<tr><td>{conn_type}</td><td>{power}</td><td>{current_type}</td></tr>"
                popup_html += "</table>"

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=420),
                icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
                tooltip=row.get("AddressInfo.Title", ""),
            ).add_to(cluster)

    folium.LayerControl(collapsed=True).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 220px;
        background-color: white; border:2px solid grey; z-index:9999;
        font-size:14px; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <b>Legenda:</b><br>
    <i class="fa fa-map-marker fa-2x" style="color:green"></i> 1 aansluiting<br>
    <i class="fa fa-map-marker fa-2x" style="color:blue"></i> 2–3 aansluitingen<br>
    <i class="fa fa-map-marker fa-2x" style="color:red"></i> ≥4 aansluitingen
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=1300, height=800, returned_objects=[])

# -------------------------
# 📊 Analyse van laadsessies (Charging_data.pkl)
# -------------------------
elif pagina == "📊 Analyse van laadsessies":
    st.title("📊 Analyse van laadsessies")
    st.markdown("Visualisaties van laadsessie data uit 'Charging_data.pkl'.")


    @st.cache_data
    def load_pickle(file_or_path):
        with open(file_or_path, 'rb') as f:
            return pickle.load(f)


    app_dir = Path(__file__).parent
    data_path = app_dir / "Charging_data.pkl"

    if not data_path.exists():
        st.error(
            f"⚠️ Bestand niet gevonden:\n{data_path}\n\nZorg dat 'Charging_data.pkl' in dezelfde map staat als dit script.")
        st.stop()

    charging = load_pickle(data_path)

    # Robustheid: zorg dat start_time datetime is
    if 'start_time' in charging.columns:
        charging['start_time'] = pd.to_datetime(charging['start_time'], errors='coerce')

    # Voor de metingen: bewaar de 'rows_before' vóór de filtering
    rows_before = len(charging)

    # Opschonen (IQR outliers op energy_delivered)
    if "energy_delivered [kWh]" in charging.columns:
        E_q75 = charging["energy_delivered [kWh]"].quantile(0.75)
        E_q25 = charging["energy_delivered [kWh]"].quantile(0.25)
        E_iqr = E_q75 - E_q25
        upper = E_q75 + 1.5 * E_iqr
        lower = E_q25 - 1.5 * E_iqr
        # Toepassen filter
        charging = charging[
            (charging["energy_delivered [kWh]"] > lower) & (charging["energy_delivered [kWh]"] < upper)].copy()

    rows_after = len(charging)
    removed = rows_before - rows_after

    # Feature engineering
    charging['charging_duration'] = pd.to_timedelta(charging.get('charging_duration', pd.NaT), errors='coerce')
    charging['date'] = charging['start_time'].dt.date
    charging['start_hour'] = charging['start_time'].dt.hour
    charging['weekday'] = charging['start_time'].dt.day_name()
    charging['month'] = charging['start_time'].dt.month_name()
    charging['is_weekend'] = charging['start_time'].dt.dayofweek >= 5  # 5=Sat, 6=Sun
    charging['charging_duration_hours'] = charging['charging_duration'].dt.total_seconds() / 3600

    # Voorkom deling door nul/infinite
    charging['avg_power_kW'] = charging.apply(
        lambda r: (r["energy_delivered [kWh]"] / r["charging_duration_hours"])
        if pd.notnull(r["energy_delivered [kWh]"]) and pd.notnull(r["charging_duration_hours"]) and r[
            "charging_duration_hours"] > 0
        else np.nan,
        axis=1
    )

    # Ordening
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December']

    # --- Figuur 1: lijn (dag/week) met selecteerbare knoppen ---
    daily = charging.groupby('date', as_index=False)['avg_power_kW'].mean().sort_values('date')
    daily['avg_power_kW_week'] = daily['avg_power_kW'].rolling(window=7, min_periods=1).mean()

    fig1 = px.line(
        daily, x='date', y=['avg_power_kW', 'avg_power_kW_week'],
        labels={'value': 'kW', 'date': 'Datum', 'variable': 'Reeks'},
        title='Gemiddelde vermogen (kW) per dag en per week'
    )
    # Trace-namen en stijl
    if len(fig1.data) >= 2:
        fig1.data[0].name = 'Gemiddeld per dag'
        fig1.data[1].name = 'Gemiddeld per week'
    fig1.update_traces(mode='lines')
    fig1.update_layout(
        title={'text': 'Gemiddelde vermogen (kW) per dag en per week', 'x': 0.5, 'xanchor': 'center'},
        updatemenus=[dict(
            buttons=[
                dict(label='Beide', method='update', args=[{'visible': [True, True]}]),
                dict(label='Gemiddeld per week', method='update', args=[{'visible': [False, True]}]),
                dict(label='Gemiddeld per dag', method='update', args=[{'visible': [True, False]}]),
            ],
            direction='down', showactive=True, x=0.0, xanchor='left', y=1.15, yanchor='top'
        )]
    )

    # --- Figuur 2: histogram uur x week met selecteerbare knoppen ---
    df2 = charging[['start_hour', 'is_weekend']].copy()
    df2['Week'] = df2['is_weekend'].map({True: 'Weekend', False: 'Werkdagen'})

    fig2 = px.histogram(
        df2, x='start_hour', color='Week', nbins=24, barmode='group',
        labels={'start_hour': 'Uur', 'Week': 'Week'},
        title='Aantal laadsessies per uur (weekdag vs weekend)'
    )
    # Zorg dat trace-namen consistent zijn (volgorde kan wisselen)
    name_map = {'Werkdagen': 'Werkdagen', 'Weekend': 'Weekend'}
    for tr in fig2.data:
        if 'Weekend' in tr.name:
            tr.name = name_map['Weekend']
        elif 'Werk' in tr.name or 'Werkdagen' in tr.name:
            tr.name = name_map['Werkdagen']

    fig2.update_xaxes(dtick=1, title='Uur van de dag')
    fig2.update_yaxes(title='Aantal laadsessies')
    fig2.update_layout(
        title={'text': 'Aantal laadsessies per uur (weekdag vs weekend)', 'x': 0.5, 'xanchor': 'center'},
        updatemenus=[dict(
            buttons=[
                dict(label='Beide', method='update', args=[{'visible': [True, True]}]),
                dict(label='Werkdagen', method='update', args=[{'visible': [True, False]}]),
                dict(label='Weekend', method='update', args=[{'visible': [False, True]}]),
            ],
            direction='down', showactive=True, x=0.0, xanchor='left', y=1.15, yanchor='top'
        )]
    )

    # --- Figuur 3: bar maand + dropdown filter (lege maanden verbergen) ---
    df3 = charging[['month', 'energy_delivered [kWh]']].copy()
    df3['month'] = pd.Categorical(df3['month'], categories=month_order, ordered=True)
    agg3 = df3.groupby('month', as_index=False)['energy_delivered [kWh]'].sum()

    # alternatief zonder lege maanden
    filtered = agg3[agg3['energy_delivered [kWh]'] > 0]

    fig3 = px.bar(
        agg3, x='month', y='energy_delivered [kWh]',
        labels={'month': 'Maand', 'energy_delivered [kWh]': 'kWh'},
        title='Totale geleverde energie per maand'
    )
    fig3.update_xaxes(tickangle=45)
    # tweede trace (zonder lege maanden), start onzichtbaar
    fig3.add_bar(
        x=filtered['month'], y=filtered['energy_delivered [kWh]'],
        name='Zonder lege maanden', visible=False
    )
    fig3.update_layout(
        title={'text': 'Totale geleverde energie per maand', 'x': 0.5, 'xanchor': 'center'},
        updatemenus=[dict(
            type='dropdown', direction='down', showactive=True,
            x=0.0, xanchor='left', y=1.15, yanchor='top',
            buttons=[
                dict(label='Toon alle maanden', method='update', args=[{'visible': [True, False]}]),
                dict(label='Verberg lege maanden', method='update', args=[{'visible': [False, True]}]),
            ]
        )]
    )

    # --- Tonen Visualisaties ---
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # --- Nieuwe Tabellen (onder de plots) ---
    st.markdown("---")
    st.markdown("## Overzicht schoonmaak-acties")

    # 1. Rijen telling metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Rijen vóór outlier-filter", rows_before)
    c2.metric("Rijen na outlier-filter", rows_after)
    c3.metric("Verwijderd (outliers)", removed)

    st.markdown("### Nieuwe (afgeleide) kolommen")

    # 2. Afgeleide kolommen tabel
    desc = pd.DataFrame([
        {"Nieuwe kolom": "date", "Omschrijving": "Datum (YYYY-MM-DD) uit start_time"},
        {"Nieuwe kolom": "start_hour", "Omschrijving": "Uur (0–23) van start_time"},
        {"Nieuwe kolom": "weekday", "Omschrijving": "Dagnaam van start_time"},
        {"Nieuwe kolom": "month", "Omschrijving": "Maandnaam van start_time"},
        {"Nieuwe kolom": "is_weekend", "Omschrijving": "Boolean: za/zo = True, anders False"},
        {"Nieuwe kolom": "charging_duration_hours", "Omschrijving": "Laadduur in uren (timedelta → uren)"},
        {"Nieuwe kolom": "avg_power_kW", "Omschrijving": "Gemiddeld vermogen = kWh / uren (filtered)"},
    ])
    st.dataframe(desc, use_container_width=True)
elif pagina == "🚗 Analyse van voertuigdata":
    # -------------------------
    # 🚗 Analyse van voertuigdata - API-versie
    # -------------------------
    st.title("🚗 Analyse van voertuigdata")

    st.caption(
        "Bron: [RDW Open Data – Elektrische voertuigen](https://opendata.rdw.nl/Voertuigen/Elektrische-voertuigen/w4rt-e856/about_data) "
        "versie SODA2 – "
        "[Endpoint](https://opendata.rdw.nl/resource/w4rt-e856.json)"
    )

    # ---------- DATA INLADEN VIA RDW API ----------
    @st.cache_data(ttl=86400)
    def load_rdw_data(limit=50000):
        """Laadt alleen de kolommen die nodig zijn voor analyses."""
        base_url = "https://opendata.rdw.nl/resource/w4rt-e856.json"
        kolommen = ["merk", "datum_eerste_toelating"]
        all_data = []
        offset = 0
        while True:
            params = {"$limit": limit, "$offset": offset, "$select": ", ".join(kolommen)}
            resp = requests.get(base_url, params=params)
            batch = resp.json()
            if not batch:
                break
            all_data.extend(batch)
            offset += limit
            if len(batch) < limit:
                break
        return pd.DataFrame(all_data)

    try:
        cars = load_rdw_data(limit=50000)
    except Exception as e:
        st.error(f"⚠️ Fout bij ophalen van RDW API: {e}")
        st.stop()

    st.success(f"✅ Data geladen: {len(cars):,} rijen en {len(cars.columns)} kolommen")

    # ---------- SCHOONMAAK ----------
    cars = cars.copy()
    if "merk" in cars.columns:
        cars["merk"] = cars["merk"].astype(str).str.strip()

    # Datumkolommen correct omzetten
    date_cols = [c for c in cars.columns if "datum" in c.lower()]
    for c in date_cols:
        cars[c] = pd.to_datetime(cars[c], errors="coerce")

    # Voeg jaar en maand toe uit datum_eerste_toelating (indien aanwezig)
    if "datum_eerste_toelating" in cars.columns:
        cars = cars.dropna(subset=["datum_eerste_toelating"])
        cars["jaar"] = cars["datum_eerste_toelating"].dt.year
        cars["maand"] = cars["datum_eerste_toelating"].dt.month
        new_columns_added = ["jaar", "maand"]

        # Filter op bouwjaar vanaf 2010
        cars = cars[cars["jaar"] >= 2010].copy()
        st.info(f"📅 Gefilterd op voertuigen vanaf 2010 — overgebleven: {len(cars):,} rijen")
    else:
        new_columns_added = []

    # ---------- VISUALISATIES ----------

    # 1️⃣ Registratie per maand per jaar
    st.subheader("Voertuigregistratie per maand per jaar")
    if "jaar" in cars.columns and "maand" in cars.columns:
        aantal_per_maand = (
            cars.groupby(["jaar", "maand"])
            .size()
            .reset_index(name="aantal_voertuigen")
            .sort_values(["jaar", "maand"])
        )
        jaren_selectie = st.multiselect(
            "Selecteer een of meerdere jaren",
            sorted(aantal_per_maand["jaar"].unique()),
            default=[aantal_per_maand["jaar"].max()],
            key="selectie_maand"
        )
        fig1 = go.Figure()
        ymax = aantal_per_maand[aantal_per_maand["jaar"].isin(jaren_selectie)]["aantal_voertuigen"].max()
        for jaar in jaren_selectie:
            df_jaar = aantal_per_maand[aantal_per_maand["jaar"] == jaar]
            fig1.add_trace(go.Scatter(
                x=df_jaar["maand"],
                y=df_jaar["aantal_voertuigen"],
                mode="lines+markers",
                name=str(jaar)
            ))
        fig1.update_layout(
            title=f"Aantal voertuigen per maand ({', '.join(map(str, jaren_selectie))})",
            xaxis_title="Maand",
            yaxis_title="Aantal voertuigen",
            plot_bgcolor="white",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"],
                gridcolor="lightgray"
            ),
            yaxis=dict(gridcolor="lightgray", range=[0, ymax]),
            legend_title="Jaar"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Geen datumgegevens beschikbaar voor maandelijkse tellingen.")

    # 2️⃣ Bar per merk
    st.subheader("Voertuigmerken per jaar")
    if all(c in cars.columns for c in ["merk", "jaar"]):
        merk_per_jaar = (
            cars.groupby(["jaar", "merk"])
            .size()
            .reset_index(name="aantal")
            .sort_values(["jaar", "aantal"], ascending=[True, False])
        )
        jaren_selectie_merk = st.multiselect(
            "Selecteer een of meerdere jaren",
            sorted(merk_per_jaar["jaar"].unique()),
            default=[max(merk_per_jaar["jaar"].unique())],
            key="selectie_merk"
        )
        fig2 = go.Figure()
        ymax = merk_per_jaar[merk_per_jaar["jaar"].isin(jaren_selectie_merk)]["aantal"].max()
        for jaar in jaren_selectie_merk:
            df_j = merk_per_jaar[merk_per_jaar["jaar"] == jaar].nlargest(20, "aantal")
            fig2.add_trace(go.Bar(
                x=df_j["merk"],
                y=df_j["aantal"],
                name=str(jaar)
            ))
        fig2.update_layout(
            title=f"Aantal voertuigen per merk ({', '.join(map(str, jaren_selectie_merk))})",
            xaxis_title="Merk",
            yaxis_title="Aantal voertuigen",
            plot_bgcolor="white",
            xaxis=dict(tickangle=-45, gridcolor="lightgray"),
            yaxis=dict(gridcolor="lightgray", range=[0, ymax]),
            legend_title="Jaar",
            barmode="group"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Kolommen 'merk' en 'jaar' zijn nodig voor deze grafiek.")
    # 4️⃣ Extra grafiek: aantal voertuigen per jaar voor geselecteerde merken
    st.subheader("Aantal voertuigen per jaar per merk (top 5)")
    merken = ['TESLA', 'VOLKSWAGEN', 'KIA', 'VOLVO', 'HYUNDAI']
    fig4 = go.Figure()
    for merk in merken:
        df_merk = cars[cars['merk'].str.upper() == merk].groupby('jaar')['merk'].count().reset_index(name='aantal')
        fig4.add_trace(go.Scatter(
            x=df_merk['jaar'],
            y=df_merk['aantal'],
            mode='lines+markers',
            name=merk
        ))
    fig4.update_layout(
        title="Aantal voertuigen per jaar per merk",
        xaxis_title="Jaar",
        yaxis_title="Aantal voertuigen",
        template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)
    # 3️⃣ Scatter aantal voertuigen totaal per jaar + Gompertz-afgeleide fit
    st.subheader("Aantal voertuigen per jaar")

    tot_per_jaar = (
        cars.groupby("jaar").size().reset_index(name="aantal").sort_values("jaar")
    )
    tot_per_jaar = tot_per_jaar[tot_per_jaar["jaar"] >= 2010]

    # --- Bereid data voor ---
    fit_df = tot_per_jaar[tot_per_jaar["aantal"] > 0].copy()
    if len(fit_df) >= 4:
        x = fit_df["jaar"].to_numpy(dtype=float)
        y = fit_df["aantal"].to_numpy(dtype=float)

        # --- Cumulatieve waarden en normalisatie ---
        cum_y = np.cumsum(y)
        x_norm = x - x.min()


        # --- Gompertz-functie voor cumulatief ---
        def gompertz_cum(x, A, B, C):
            return A * np.exp(-B * np.exp(-C * x))


        # --- Fit uitvoeren ---
        from scipy.optimize import curve_fit

        p0 = [cum_y.max() * 1.1, 1.0, 0.2]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        popt, _ = curve_fit(gompertz_cum, x_norm, cum_y, p0=p0, bounds=bounds, maxfev=20000)
        A_cum, B_cum, C_cum = popt


        # --- Afgeleide van cumulatieve Gompertz (jaarlijkse verkopen) ---
        def gompertz_deriv_from_cum(x, A, B, C):
            return A * B * C * np.exp(-C * x) * np.exp(-B * np.exp(-C * x))


        # --- Voorspelling ---
        y_pred = gompertz_deriv_from_cum(x_norm, A_cum, B_cum, C_cum)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # --- Fitlijn genereren ---
        x_line = np.linspace(x_norm.min(), x_norm.max() + 5, 300)
        y_line = gompertz_deriv_from_cum(x_line, A_cum, B_cum, C_cum)

        # --- Plot maken ---
        fig3 = go.Figure()

        # Punten (neutraal grijs)
        fig3.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Waarnemingen",
            marker=dict(size=8, color="rgba(80,80,80,0.8)", line=dict(width=0)),
        ))

        # Fitlijn (zwarte stippellijn)
        fig3.add_trace(go.Scatter(
            x=x_line + x.min(),
            y=y_line,
            mode="lines",
            name="Gompertz-afgeleide fit",
            line=dict(color="black", width=3, dash="dash"),
        ))

        # --- Layout ---
        fig3.update_layout(
            title="Aantal elektrische voertuigen per jaar",
            xaxis_title="Jaar",
            yaxis_title="Aantal voertuigen",
            plot_bgcolor="white",
            xaxis=dict(gridcolor="lightgray", tickmode="linear"),
            yaxis=dict(gridcolor="lightgray"),
            legend=dict(
                title="Legenda",
                yanchor="top", y=1,
                xanchor="left", x=1.02
            ),
            title_x=0.5
        )

        # Toon grafiek + statistieken
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"📊 R² van de Gompertz-afgeleide fit: {r_squared:.4f}")
        st.caption(f"📈 Parameters: A={A_cum:.0f}, B={B_cum:.3f}, C={C_cum:.3f}")
    else:
        st.warning("Onvoldoende data om de Gompertz-fit te berekenen.")

# -------------------------
# 🚗 Analyse van voertuigdata (oud)
# -------------------------
elif pagina == "🚗👴 Analyse van voertuigdata (oud)":
    st.title("🚗👴 Analyse van voertuigdata (oude versie)")


    @st.cache_data
    def load_pickle(file_or_path):
        with open(file_or_path, 'rb') as f:
            return pickle.load(f)


    app_dir = Path(__file__).parent
    data_path = app_dir / "cars.pkl"

    if not data_path.exists():
        st.error(f"⚠️ Bestand niet gevonden:\n{data_path}\n\nZorg dat 'cars.pkl' in dezelfde map staat als dit script.")
        st.stop()

    cars = load_pickle(data_path)

    # ---------- SCHOONMAAK ----------
    cars = cars.copy()
    # 1) Typecasts & trim
    if 'merk' in cars.columns:
        cars['merk'] = cars['merk'].astype(str).str.strip()

    typecasts_applied = {
        "kenteken": "string",
        "massa_ledig_voertuig": "float64",
        "massa_rijklaar": "float64",
        "wielbasis": "float64",
        "catalogusprijs": "float64",
        "aantal_deuren": "Int64",
        "aantal_wielen": "Int64",
        "lengte": "float64",
        "breedte": "float64",
        "hoogte_voertuig": "float64",
    }
    for c, t in typecasts_applied.items():
        if c in cars.columns:
            if t.lower() == "string":
                cars[c] = cars[c].astype("string")
            else:
                cars[c] = pd.to_numeric(cars[c], errors="coerce").astype(t, errors="ignore")

    # 2) Drop kolommen
    dropped_columns = []
    if "bruto_bpm" in cars.columns:
        cars = cars.drop("bruto_bpm", axis=1)
        dropped_columns.append("bruto_bpm")

    # 3) NaN-imputatie per type kolom (mediaan / mediane datum / modus)
    overzicht = []
    for col in cars.columns:
        na_before = cars[col].isna().sum()
        ingevulde_waarde = None
        methode = "geen"

        if pd.api.types.is_numeric_dtype(cars[col]):
            median_value = cars[col].median()
            cars[col] = cars[col].fillna(median_value)
            ingevulde_waarde = median_value
            methode = "mediaan"

        elif pd.api.types.is_datetime64_any_dtype(cars[col]) or (
                cars[col].dtype == "object" and "datum" in col.lower()):
            cars[col] = pd.to_datetime(cars[col], errors="coerce")
            median_date = cars[col].median()
            cars[col] = cars[col].fillna(median_date)
            ingevulde_waarde = median_date
            methode = "mediane datum"

        elif pd.api.types.is_object_dtype(cars[col]) or pd.api.types.is_string_dtype(cars[col]):
            mode_series = cars[col].mode(dropna=True)
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                cars[col] = cars[col].fillna(mode_value)
                ingevulde_waarde = mode_value
                methode = "modus"

        na_after = cars[col].isna().sum()


        # Optioneel: nette representatie voor datetime
        def _fmt(v):
            if isinstance(v, pd.Timestamp):
                return v.strftime("%Y-%m-%d")
            return v


        overzicht.append({
            "Kolom": col,
            "Type": str(cars[col].dtype),
            "NaN vóór invullen": int(na_before),
            "Methode invullen": methode,
            # "NaN ná invullen": int(na_after), # <-- DEZE IS NU VERWIJDERD
        })

    overzicht_df = pd.DataFrame(overzicht).sort_values("NaN vóór invullen", ascending=False).reset_index(drop=True)

    # 4) Outlier-filter (IQR 1.5x) op alle numerieke kolommen (behalve aantal_wielen)
    numeric_cols = cars.select_dtypes(include=['number']).columns.tolist()
    cols_for_outliers = [c for c in numeric_cols if c != "aantal_wielen"]
    mask = pd.Series(True, index=cars.index)
    for col in cols_for_outliers:
        s = pd.to_numeric(cars[col], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty or s.nunique() < 2:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask &= (cars[col] >= lower) & (cars[col] <= upper)

    rows_before = len(cars)
    cars_clean = cars[mask].copy()
    rows_after = len(cars_clean)
    outliers_removed = rows_before - rows_after

    # 5) Datum + nieuwe kolommen
    cars_clean["datum_eerste_toelating"] = pd.to_datetime(cars_clean.get("datum_eerste_toelating"), format="%Y%m%d",
                                                          errors="coerce")
    cars_clean = cars_clean.dropna(subset=["datum_eerste_toelating"])
    cars_clean["jaar"] = cars_clean["datum_eerste_toelating"].dt.year
    cars_clean["maand"] = cars_clean["datum_eerste_toelating"].dt.month
    new_columns_added = ["jaar", "maand"]


    # ---------- Grafieken (met cars_clean) - NU BOVEN ----------
    st.markdown("## Visualisaties") # Grotere kop

    # (1) Lijndiagram per jaar met dropdown
    st.subheader("Voertuigregistratie per maand") # Subkop
    aantal_per_maand = (
        cars_clean.groupby(["jaar", "maand"])
        .size()
        .reset_index(name="aantal_voertuigen")
        .sort_values(["jaar", "maand"])
    )
    fig1 = go.Figure()
    jaren = aantal_per_maand["jaar"].unique()
    if len(jaren) > 0:
        for jaar in jaren:
            df_jaar = aantal_per_maand[aantal_per_maand["jaar"] == jaar]
            fig1.add_trace(go.Scatter(
                x=df_jaar["maand"], y=df_jaar["aantal_voertuigen"],
                mode="lines+markers", name=str(jaar),
                visible=(jaar == jaren[-1])  # start met laatste jaar
            ))

        dropdown_knoppen = []
        for i, jaar in enumerate(jaren):
            vis = [False] * len(jaren);
            vis[i] = True
            dropdown_knoppen.append(dict(label=str(jaar), method="update", args=[{"visible": vis}]))

        fig1.update_layout(
            title={'text': f"Aantal voertuigen per maand in {jaren[-1]}", 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Maand", yaxis_title="Aantal voertuigen",
            updatemenus=[dict(
                type='dropdown', direction='down', showactive=True,
                active=len(jaren) - 1, buttons=dropdown_knoppen,
                x=0.0, xanchor="left", y=1.15, yanchor="top",
            )],
            plot_bgcolor="white",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"],
                gridcolor="lightgray"
            ),
            yaxis=dict(gridcolor="lightgray")
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Geen geldige data gevonden voor voertuigregistratie.")

    # (2) Bar per merk
    st.subheader("Voertuigmerken") # Subkop
    if 'merk' in cars_clean.columns:
        merk_counts = cars_clean["merk"].value_counts().reset_index()
        merk_counts.columns = ["merk", "aantal"]
        fig2 = px.bar(
            merk_counts, x="merk", y="aantal",
            title="Aantal voertuigen per merk", text="aantal", color="merk"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            title={'text': 'Aantal voertuigen per merk', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Merk", yaxis_title="Aantal",
            showlegend=False, plot_bgcolor="white",
            xaxis=dict(gridcolor="lightgray"), yaxis=dict(gridcolor="lightgray")
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Kolom 'merk' niet beschikbaar voor deze visualisatie.")

    # (3) Scatter + toggle regressielijn
    st.subheader("Wielbasis vs. Massa Rijklaar") # Subkop
    if "wielbasis" in cars_clean.columns and "massa_rijklaar" in cars_clean.columns and "merk" in cars_clean.columns:
        cars_clean["wielbasis"] = pd.to_numeric(cars_clean["wielbasis"], errors="coerce")
        cars_clean["massa_rijklaar"] = pd.to_numeric(cars_clean["massa_rijklaar"], errors="coerce")

        fig3 = px.scatter(
            cars_clean, x="wielbasis", y="massa_rijklaar", color="merk",
            title="Relatie tussen wielbasis en massa rijklaar per merk",
            labels={"wielbasis": "Wielbasis (cm)", "massa_rijklaar": "Massa rijklaar (kg)", "merk": "Merk"},
            hover_data=["handelsbenaming", "inrichting", "datum_eerste_toelating"]
        )

        mask = cars_clean["wielbasis"].notna() & cars_clean["massa_rijklaar"].notna()
        if mask.any():
            x = cars_clean.loc[mask, "wielbasis"].to_numpy()
            y = cars_clean.loc[mask, "massa_rijklaar"].to_numpy()
            if len(x) > 1:  # Zorg voor minimaal 2 punten voor polyfit
                coef = np.polyfit(x, y, 1)
                poly_fn = np.poly1d(coef)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = poly_fn(x_line)
            else:
                x_line = np.array([]);
                y_line = np.array([])
        else:
            x_line = np.array([]);
            y_line = np.array([])

        num_scatter = len(fig3.data)
        fig3.add_trace(go.Scatter(
            x=x_line, y=y_line, mode="lines", name="Trendlijn",
            line=dict(width=2, dash="dash"), visible=True
        ))
        fig3.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode="markers"))
        fig3.update_layout(
            legend_title_text="Merk", plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgray", title="Wielbasis (cm)"),
            yaxis=dict(showgrid=True, gridcolor="lightgray", title="Massa rijklaar (kg)"),
            title={'text': 'Relatie tussen wielbasis en massa rijklaar per merk', 'x': 0.5, 'xanchor': 'center'},
            updatemenus=[dict(
                type='dropdown', direction='down', showactive=True,
                x=0.0, xanchor='left', y=1.15, yanchor='top',
                buttons=[
                    dict(label='Met regressielijn', method='update', args=[{'visible': [True] * num_scatter + [True]}]),
                    dict(label='Zonder regressielijn', method='update',
                         args=[{'visible': [True] * num_scatter + [False]}]),
                ],
            )]
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(
            "Niet alle benodigde kolommen ('wielbasis', 'massa_rijklaar', 'merk') zijn beschikbaar voor deze visualisatie.")


    # ---------- UI: overzicht schoonmaak, nu ONDER ----------
    st.markdown("---")
    st.markdown("## Overzicht schoonmaak-acties") # Grotere kop

    # Tabel voor gedropte en nieuwe kolommen
    drop_new_df = pd.DataFrame({
        "Gedropte kolommen": [", ".join(dropped_columns) if dropped_columns else "—"],
        "Nieuwe kolommen toegevoegd": [", ".join(new_columns_added)]
    })
    st.dataframe(drop_new_df.T.rename(columns={0: "Waarden"}), use_container_width=True)


    st.markdown("### NaN-invulling per kolom") # Subkop
    # De overzicht_df heeft nu 3 kolommen
    st.dataframe(overzicht_df.head(20), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Rijen vóór outlier-filter", rows_before)
    m2.metric("Rijen na outlier-filter", rows_after)
    m3.metric("Verwijderd (outliers)", outliers_removed)



