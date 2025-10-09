# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, Fullscreen
from shapely.geometry import Point
import geopandas as gpd
from streamlit_folium import st_folium
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# import matplotlib.pyplot as plt # Niet nodig voor Streamlit output

# Stel de basisconfiguratie van de pagina in
st.set_page_config(page_title="Laadpalen, Laadsessies & Voertuigdata Nederland", layout="wide")

# De sidebar voor navigatie
st.sidebar.title("🔀 Navigatie")
pagina = st.sidebar.radio(
    "Kies een onderdeel:",
    ["🗺️ Kaart van laadpalen", "📊 Analyse van laadsessies", "🚗 Analyse van voertuigdata"]
)

# -------------------------
# 🗺️ Kaart van laadpalen
# -------------------------
if pagina == "🗺️ Kaart van laadpalen":
    st.title("🚗 Laadpalenkaart Nederland")
    st.markdown(
        "Selecteer één provincie of 'Alle provincies'. De kaart zoomt automatisch in. Data is gecachet voor snellere herlaad.")


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

    # Opschonen (IQR outliers op energy_delivered)
    if "energy_delivered [kWh]" in charging.columns:
        E_q75 = charging["energy_delivered [kWh]"].quantile(0.75)
        E_q25 = charging["energy_delivered [kWh]"].quantile(0.25)
        E_iqr = E_q75 - E_q25
        upper = E_q75 + 1.5 * E_iqr
        lower = E_q25 - 1.5 * E_iqr
        charging = charging[(charging["energy_delivered [kWh]"] > lower) & (charging["energy_delivered [kWh]"] < upper)]

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
    for tr in fig2.data:
        if 'Weekend' in tr.name:
            tr.name = 'Weekend'
        elif 'Werk' in tr.name or 'Werkdagen' in tr.name:
            tr.name = 'Werkdagen'
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

    # --- Tonen ---
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)


# -------------------------
# 🚗 Analyse van voertuigdata (cars.pkl)
# -------------------------
elif pagina == "🚗 Analyse van voertuigdata":
    st.title("🚗 Analyse van voertuigdata")
    st.markdown("Visualisaties van voertuigdata uit 'cars.pkl'.")


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

    # --- Data opschonen en voorbereiden (uit de tweede code) ---
    # os.chdir('C:\\school\\Datascience\\laadpalen_elektrisch_vervoer') # Niet nodig in Streamlit

    # Typeconversies (kolommen in de tweede code)
    if 'merk' in cars.columns:
        cars['merk'] = cars['merk'].astype(str).str.strip()
    if "kenteken" in cars.columns:
        cars["kenteken"] = cars["kenteken"].astype("string")
    if 'massa_ledig_voertuig' in cars.columns:
        cars['massa_ledig_voertuig'] = cars['massa_ledig_voertuig'].astype('float64')
    if "massa_rijklaar" in cars.columns:
        cars["massa_rijklaar"] = cars["massa_rijklaar"].astype("float64")
    if "wielbasis" in cars.columns:
        cars["wielbasis"] = cars["wielbasis"].astype("float64")
    if 'catalogusprijs' in cars.columns:
        cars['catalogusprijs'] = cars['catalogusprijs'].astype('float64')
    if 'aantal_deuren' in cars.columns:
        cars['aantal_deuren'] = cars['aantal_deuren'].astype('Int64')
    if 'aantal_wielen' in cars.columns:
        cars['aantal_wielen'] = cars['aantal_wielen'].astype('Int64')
    if "lengte" in cars.columns:
        cars["lengte"] = cars["lengte"].astype("float64")
    if "breedte" in cars.columns:
        cars["breedte"] = cars["breedte"].astype("float64")
    if "hoogte_voertuig" in cars.columns:
        cars["hoogte_voertuig"] = cars["hoogte_voertuig"].astype("float64")

    # Drop kolom
    if 'bruto_bpm' in cars.columns:
        cars = cars.drop('bruto_bpm', axis=1)

    # Imputatie van NaN-waarden (vereenvoudigd voor de app)
    # Dit stuk wordt weggelaten of vervangen door een snelle fillna,
    # want de imputatie-analyse is meer voor data-cleaning scripts dan de app-display.

    # We vullen de NaNs nu in met de mediaan/modus, zoals in de originele code
    for col in cars.columns:
        if cars[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(cars[col]):
                cars[col] = cars[col].fillna(cars[col].median())
            elif pd.api.types.is_datetime64_any_dtype(cars[col]) or (
                    "datum" in col.lower() and cars[col].dtype == "object"):
                cars[col] = pd.to_datetime(cars[col], errors="coerce").fillna(
                    pd.to_datetime(cars[col], errors="coerce").median())
            elif pd.api.types.is_object_dtype(cars[col]) or pd.api.types.is_string_dtype(cars[col]):
                if not cars[col].mode(dropna=True).empty:
                    cars[col] = cars[col].fillna(cars[col].mode(dropna=True)[0])

    # Outlier verwijdering (IQR)
    numeric_cols = cars.select_dtypes(include=['number']).columns.tolist()
    cols_to_clean = [c for c in numeric_cols if c != "aantal_wielen" and c in cars.columns]  # check of kolom bestaat

    mask = pd.Series(True, index=cars.index)
    for col in cols_to_clean:
        s = cars[col]
        s = s[np.isfinite(s)]
        if s.empty or s.nunique() < 2:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= (cars[col] >= lower) & (cars[col] <= upper)
    cars_clean = cars[mask].copy()

    st.header("1. Voertuigregistratie over tijd")

    # --- Voertuigregistratie per jaar en maand ---
    cars_clean["datum_eerste_toelating"] = pd.to_datetime(cars_clean["datum_eerste_toelating"], format="%Y%m%d",
                                                          errors="coerce")
    cars_clean = cars_clean.dropna(subset=["datum_eerste_toelating"])

    cars_clean["jaar"] = cars_clean["datum_eerste_toelating"].dt.year
    cars_clean["maand"] = cars_clean["datum_eerste_toelating"].dt.month

    aantal_per_maand = (
        cars_clean.groupby(["jaar", "maand"])
        .size()
        .reset_index(name="aantal_voertuigen")
        .sort_values(["jaar", "maand"])
    )

    fig_reg = go.Figure()
    jaren = aantal_per_maand["jaar"].unique()

    if len(jaren) > 0:
        for jaar in jaren:
            df_jaar = aantal_per_maand[aantal_per_maand["jaar"] == jaar]
            fig_reg.add_trace(
                go.Scatter(
                    x=df_jaar["maand"],
                    y=df_jaar["aantal_voertuigen"],
                    mode="lines+markers",
                    name=str(jaar),
                    visible=(jaar == jaren[-1])
                )
            )

        dropdown_knoppen = []
        for i, jaar in enumerate(jaren):
            visible = [False] * len(jaren)
            visible[i] = True
            knop = dict(
                label=str(jaar),
                method="update",
                args=[{"visible": visible},
                      {"title": f"Aantal voertuigen per maand in {jaar}"}]
            )
            dropdown_knoppen.append(knop)

        fig_reg.update_layout(
            title={'text': f"Aantal voertuigen per maand in {jaren[-1]}", 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Maand",
            yaxis_title="Aantal voertuigen",
            updatemenus=[
                dict(
                    active=len(jaren) - 1,
                    buttons=dropdown_knoppen,
                    x=1.1,
                    y=1.15,
                    xanchor="right",
                    yanchor="top"
                )
            ],
            plot_bgcolor="white",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"],
                gridcolor="lightgray"
            ),
            yaxis=dict(gridcolor="lightgray")
        )
        st.plotly_chart(fig_reg, use_container_width=True)
    else:
        st.info("Geen geldige data gevonden voor voertuigregistratie.")

    st.header("2. Distributie van voertuigmerken")

    # --- Aantal voertuigen per merk ---
    if 'merk' in cars_clean.columns:
        merk_counts = cars_clean["merk"].value_counts().reset_index()
        merk_counts.columns = ["merk", "aantal"]

        fig_merk = px.bar(
            merk_counts,
            x="merk",
            y="aantal",
            title="Aantal voertuigen per merk",
            text="aantal",
            color="merk"
        )

        fig_merk.update_traces(textposition="outside")
        fig_merk.update_layout(
            title={'text': 'Aantal voertuigen per merk', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Merk",
            yaxis_title="Aantal",
            showlegend=False
        )
        st.plotly_chart(fig_merk, use_container_width=True)
    else:
        st.info("Kolom 'merk' niet gevonden in de data.")

    st.header("3. Wielbasis versus Rijklaar Massa")

    # --- Scatterplot: Wielbasis vs Massa rijklaar ---
    if "wielbasis" in cars_clean.columns and "massa_rijklaar" in cars_clean.columns and "merk" in cars_clean.columns:
        # Zorg dat de kolommen numeriek zijn (opnieuw na cleaning)
        cars_clean["wielbasis"] = pd.to_numeric(cars_clean["wielbasis"], errors="coerce")
        cars_clean["massa_rijklaar"] = pd.to_numeric(cars_clean["massa_rijklaar"], errors="coerce")

        fig_scatter = px.scatter(
            cars_clean,
            x="wielbasis",
            y="massa_rijklaar",
            color="merk",
            title="Relatie tussen wielbasis en massa rijklaar per merk",
            labels={
                "wielbasis": "Wielbasis (cm)",
                "massa_rijklaar": "Massa rijklaar (kg)",
                "merk": "Merk"
            },
            hover_data=["handelsbenaming", "inrichting", "datum_eerste_toelating"]
        )

        # Regressielijn over alle punten
        mask = cars_clean["wielbasis"].notna() & cars_clean["massa_rijklaar"].notna()
        x = cars_clean.loc[mask, "wielbasis"].to_numpy()
        y = cars_clean.loc[mask, "massa_rijklaar"].to_numpy()

        if len(x) > 1:
            coef = np.polyfit(x, y, 1)
            poly_fn = np.poly1d(coef)

            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = poly_fn(x_line)

            num_scatter = len(fig_scatter.data)

            # Voeg regressielijn toe (zichtbaar bij start)
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trendlijn",
                line=dict(width=2, dash="dash"),
                visible=True
            ))

            # Stijl en dropdown menu
            fig_scatter.update_traces(marker=dict(size=10, opacity=0.7), selector=dict(mode="markers"))
            fig_scatter.update_layout(
                legend_title_text="Merk",
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="lightgray", title="Wielbasis (cm)"),
                yaxis=dict(showgrid=True, gridcolor="lightgray", title="Massa rijklaar (kg)"),
                title={'text': 'Relatie tussen wielbasis en massa rijklaar per merk', 'x': 0.5, 'xanchor': 'center'},
                updatemenus=[dict(
                    type='dropdown',
                    direction='down',
                    showactive=True,
                    x=-0.3, xanchor='left',
                    y=1.15, yanchor='top',
                    buttons=[
                        dict(
                            label='Met regressielijn',
                            method='update',
                            args=[{'visible': [True] * num_scatter + [True]}]
                        ),
                        dict(
                            label='Zonder regressielijn',
                            method='update',
                            args=[{'visible': [True] * num_scatter + [False]}]
                        ),
                    ]
                )]
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Onvoldoende data beschikbaar om de scatterplot en trendlijn te tonen.")
    else:
        st.info("Niet alle benodigde kolommen ('wielbasis', 'massa_rijklaar', 'merk') zijn aanwezig in de data.")