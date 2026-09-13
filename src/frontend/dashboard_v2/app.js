/* Hurricane Florence Dashboard V2 — static interactive frontend. */
(() => {
  "use strict";

  const SOURCE_COLORS = {
    ml: "#2f80ed",
    physics: "#f2994a",
    observed: "#27ae60",
  };
  const SOURCE_ORDER = ["ml", "physics", "observed"];

  let bundle = null;
  let contextGeo = {};
  let impactVectorGeo = { roads: {} };
  let singleMap = null;
  let leftMap = null;
  let rightMap = null;
  let singleRasterLayers = [];
  let leftRasterLayers = [];
  let rightRasterLayers = [];
  let mapSyncGuard = false;
  let mapViewState = null;
  let initialMapFitDone = false;
  let playTimer = null;
  const playFadeFrames = new Set();
  let isPlaying = false;
  const PLAY_INTERVAL_MS = 1650;
  const PLAY_FADE_MS = 650;
  let activeWorkspaceView = "map";

  let hqChart = null;
  let hydraulicChart = null;
  let rainfallChart = null;
  let landcoverChart = null;
  let miniCharts = [];

  const $ = (id) => document.getElementById(id);
  const fmt = (value, decimals = 2) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toLocaleString(undefined, {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  function setOptions(select, options, current = null) {
    const prev = current ?? select.value;
    select.innerHTML = "";
    options.forEach(({ value, label }) => {
      const opt = document.createElement("option");
      opt.value = value;
      opt.textContent = label;
      select.appendChild(opt);
    });
    if (options.some((o) => o.value === prev)) select.value = prev;
  }

  function sourceLabel(key) {
    return bundle?.sources?.[key]?.short || key;
  }

  function hydrateControls() {
    setOptions($("dateSelect"), bundle.dates.map((d) => ({ value: d, label: d })));
    // Primary/secondary source controls are intentionally limited to the three
    // scientific sources. Impact assessment is a map product, never a source.
    const sourceOptions = SOURCE_ORDER
      .filter((k) => ["ml", "physics", "observed"].includes(k) && bundle.sources[k])
      .map((k) => ({ value: k, label: bundle.sources[k].label }));
    setOptions($("primarySource"), sourceOptions, "physics");
    setOptions($("secondarySource"), sourceOptions, "observed");

    const defaultDate = bundle.dates.includes("2018-09-14")
      ? "2018-09-14"
      : bundle.dates[0] || bundle.map_dates?.[0];
    $("dateSelect").value = defaultDate;

    setOptions(
      $("impactMapLayer"),
      (bundle.impact?.map_layers || []).map((m) => ({ value: m.key, label: m.label })),
      "flood_area"
    );

    ["dateSelect", "primarySource", "secondarySource", "layerSelect", "compareMode"].forEach((id) => {
      $(id).addEventListener("change", handleMapControlChange);
    });
    $("primarySource").addEventListener("change", () => {
      ensureDifferentSecondary();
      updateHydraulics();
    });
    $("dateSelect").addEventListener("change", () => {
      updateHydraulics();
      updateRainfallSummary();
    });
    $("impactMapLayer").addEventListener("change", () => {
      captureVisibleMapView();
      $("layerSelect").value = "impact";
      activateMapView();
      handleMapControlChange();
    });

    ["toggleWatershed", "toggleAoi", "toggleMainstem", "toggleGauge"].forEach((id) => {
      $(id).addEventListener("change", applyContextVisibility);
    });
    $("fitWatershedBtn").addEventListener("click", fitUpstreamWatershed);
    $("playDatesBtn").addEventListener("click", toggleDatePlayback);
    $("showMapBtn").addEventListener("click", activateMapView);
    const opacity = $("overlayOpacity");
    if (opacity) {
      opacity.addEventListener("input", () => {
        updateOpacityLabel();
        applyOverlayOpacity();
      });
      updateOpacityLabel();
    }

    document.querySelectorAll(".graph-tab").forEach((button) => {
      button.addEventListener("click", () => activateGraphView(button.dataset.graph));
    });
  }

  function ensureDifferentSecondary() {
    const primary = $("primarySource").value;
    const secondary = $("secondarySource").value;
    if (primary !== secondary) return;
    const alt = SOURCE_ORDER.find((k) => bundle.sources[k] && k !== primary);
    if (alt) $("secondarySource").value = alt;
  }

  function layerSupportsComparison(layer) {
    return ["extent", "depth", "wse", "impact"].includes(layer);
  }

  function handleMapControlChange() {
    captureVisibleMapView();
    const layer = $("layerSelect").value;
    const compare = layerSupportsComparison(layer);
    $("compareMode").disabled = !compare;
    $("secondarySource").disabled = !compare;
    $("secondaryWrap").classList.toggle("hidden", !compare);
    if (!compare) $("compareMode").value = "single";
    ensureDifferentSecondary();
    renderMap();
    updateHydraulics();
    updateRainfallSummary();
  }

  function activateMapView() {
    stopDatePlayback(false);
    activeWorkspaceView = "map";
    document.querySelector(".view-stage")?.classList.remove("analysis-mode");
    $("workspace-map").classList.remove("hidden");
    $("graphWorkspace").classList.add("hidden");
    $("showMapBtn").classList.add("active");
    document.querySelectorAll(".graph-tab").forEach((b) => b.classList.remove("active"));
    ["hydraulics", "rainfall", "impact", "landcover"].forEach((key) => {
      const el = $(`graph-${key}`);
      if (el) el.classList.add("hidden");
    });
    setTimeout(() => {
      [singleMap, leftMap, rightMap].forEach((m) => {
        try { m?.invalidateSize(); } catch (_) {}
      });
      const map = visibleMap();
      if (map) restoreMapView(map);
    }, 0);
  }

  function activateGraphView(name) {
    stopDatePlayback(false);
    captureVisibleMapView();
    activeWorkspaceView = name;
    document.querySelector(".view-stage")?.classList.add("analysis-mode");
    $("workspace-map").classList.add("hidden");
    $("graphWorkspace").classList.remove("hidden");
    $("showMapBtn").classList.remove("active");
    document.querySelectorAll(".graph-tab").forEach((b) => b.classList.toggle("active", b.dataset.graph === name));
    ["hydraulics", "rainfall", "impact", "landcover"].forEach((key) => {
      const el = $(`graph-${key}`);
      if (!el) return;
      el.classList.toggle("hidden", key !== name);
    });
    // Chart.js needs a resize after a canvas becomes visible.
    setTimeout(() => {
      [hqChart, hydraulicChart, rainfallChart, landcoverChart, ...miniCharts].forEach((c) => {
        try { c?.resize(); } catch (_) {}
      });
    }, 0);
  }

  function updatePlayUi() {
    const btn = $("playDatesBtn");
    if (!btn) return;
    btn.textContent = isPlaying ? "❚❚ Pause" : "▶ Play 14–23 Sep";
    btn.setAttribute("aria-pressed", isPlaying ? "true" : "false");
    btn.classList.toggle("active", isPlaying);
    const status = $("playStatus");
    if (status) {
      status.textContent = isPlaying
        ? `Playing ${$("dateSelect").value} · smooth crossfade · viewport locked.`
        : "Play crossfades 14–23 Sep sequentially and preserves the current map center/zoom.";
    }
  }

  function stopDatePlayback(updateUi = true) {
    if (playTimer) window.clearInterval(playTimer);
    playTimer = null;
    playFadeFrames.forEach((id) => window.cancelAnimationFrame(id));
    playFadeFrames.clear();
    isPlaying = false;
    const badge = $("playDateBadge");
    if (badge) badge.classList.add("hidden");
    if (updateUi) updatePlayUi();
  }

  function advancePlaybackDate() {
    const dates = bundle?.dates || [];
    if (!dates.length) return stopDatePlayback();
    const select = $("dateSelect");
    let idx = dates.indexOf(select.value);
    if (idx < 0) idx = 0;
    if (idx >= dates.length - 1) {
      stopDatePlayback();
      return;
    }
    select.value = dates[idx + 1];
    showPlayDateBadge(select.value);
    select.dispatchEvent(new Event("change", { bubbles: true }));
    updatePlayUi();
  }

  function toggleDatePlayback() {
    if (isPlaying) {
      stopDatePlayback();
      return;
    }
    activateMapView();
    const dates = bundle?.dates || [];
    if (!dates.length) return;
    const select = $("dateSelect");
    if (dates.indexOf(select.value) >= dates.length - 1) {
      select.value = dates[0];
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
    isPlaying = true;
    showPlayDateBadge(select.value);
    updatePlayUi();
    playTimer = window.setInterval(advancePlaybackDate, PLAY_INTERVAL_MS);
  }

  function currentOverlayOpacity() {
    const el = $("overlayOpacity");
    if (!el) return 0.78;
    const value = Number(el.value);
    if (!Number.isFinite(value)) return 0.78;
    return Math.max(0, Math.min(1, value / 100));
  }

  function updateOpacityLabel() {
    const out = $("opacityValue");
    if (out) out.textContent = `${Math.round(currentOverlayOpacity() * 100)}%`;
  }

  function layerTargetOpacity(layer) {
    const scale = Number(layer?._dashboardOpacityScale ?? 1);
    return Math.max(0, Math.min(1, currentOverlayOpacity() * (Number.isFinite(scale) ? scale : 1)));
  }

  function applyOverlayOpacity() {
    [singleRasterLayers, leftRasterLayers, rightRasterLayers].forEach((layers) => {
      layers.forEach((layer) => {
        try { layer?.setOpacity?.(layerTargetOpacity(layer)); } catch (_) {}
      });
    });
  }

  function showPlayDateBadge(date) {
    const badge = $("playDateBadge");
    if (!badge) return;
    const parsed = new Date(`${date}T00:00:00Z`);
    const text = Number.isNaN(parsed.getTime())
      ? date
      : parsed.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
    badge.textContent = text;
    badge.classList.remove("hidden", "play-date-pop");
    // Restart the CSS animation every frame change.
    void badge.offsetWidth;
    badge.classList.add("play-date-pop");
  }

  async function loadContextGeo() {
    const refs = bundle.map?.geo || {};
    const entries = await Promise.all(
      Object.entries(refs).map(async ([key, path]) => {
        try {
          const r = await fetch(path);
          if (!r.ok) throw new Error(`${r.status}`);
          return [key, await r.json()];
        } catch (e) {
          console.warn(`Could not load ${key} GeoJSON`, e);
          return [key, null];
        }
      })
    );
    contextGeo = Object.fromEntries(entries.filter(([, v]) => v));
  }


  async function loadImpactVectors() {
    impactVectorGeo = { roads: {} };
    const refs = bundle.map?.impact_vectors || {};
    const jobs = [];

    SOURCE_ORDER.forEach((source) => {
      const byDate = refs?.[source]?.roads || {};
      impactVectorGeo.roads[source] = {};

      Object.entries(byDate).forEach(([date, spec]) => {
        if (!spec?.path) return;
        jobs.push((async () => {
          try {
            const response = await fetch(spec.path, { cache: "no-store" });
            if (!response.ok) throw new Error(`${response.status}`);
            impactVectorGeo.roads[source][date] = await response.json();
          } catch (error) {
            console.warn(`Could not load impacted-road vector for ${source} ${date}`, error);
          }
        })());
      });
    });

    await Promise.all(jobs);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function clearImpactVectorLayers(map) {
    if (!map?._impactVectorLayers) return;
    map._impactVectorLayers.forEach((layer) => {
      try { map.removeLayer(layer); } catch (_) {}
    });
    map._impactVectorLayers = [];
  }

  function impactedRoadGeo(source, date) {
    return impactVectorGeo?.roads?.[source]?.[date] || null;
  }

  function hasImpactedRoadVector(source, date) {
    return Boolean(impactedRoadGeo(source, date)?.features?.length);
  }

  function impactedRoadLengthKm(source, date) {
    const bundleValue = Number(bundle.impact?.daily_roads?.[source]?.[date]?.flooded_road_length_km);
    if (Number.isFinite(bundleValue)) return bundleValue;

    const geo = impactedRoadGeo(source, date);
    if (!geo?.features?.length) return 0;
    return geo.features.reduce((sum, feature) => {
      const value = Number(feature?.properties?.segment_length_km);
      return sum + (Number.isFinite(value) ? value : 0);
    }, 0);
  }

  function addImpactedRoadVector(map, source, date) {
    const geo = impactedRoadGeo(source, date);
    if (!map || !geo?.features?.length) return null;
    map._impactVectorLayers = map._impactVectorLayers || [];

    const centerColor = SOURCE_COLORS[source] || "#b42318";
    const styleBase = {
      pane: "impactVectorPane",
      lineCap: "round",
      lineJoin: "round",
    };

    // White casing keeps road lines legible above flood fill and OSM tiles.
    const casing = L.geoJSON(geo, {
      pane: "impactVectorPane",
      interactive: false,
      style: {
        ...styleBase,
        color: "#ffffff",
        weight: 7,
        opacity: 0.96,
      },
    }).addTo(map);

    const center = L.geoJSON(geo, {
      pane: "impactVectorPane",
      style: {
        ...styleBase,
        color: centerColor,
        weight: 3.5,
        opacity: 1.0,
      },
      onEachFeature: (feature, layer) => {
        const props = feature.properties || {};
        const name = props.name || props.ref || props.highway || "Road segment";
        const length = Number(props.segment_length_km);
        const detail = Number.isFinite(length) ? `${fmt(length, 3)} km` : "—";
        layer.bindPopup(
          `<strong>${escapeHtml(name)}</strong><br>` +
          `Impacted segment: ${detail}<br>` +
          `Date: ${escapeHtml(date)}<br>` +
          `Source: ${escapeHtml(sourceLabel(source))}`
        );
      },
    }).addTo(map);

    map._impactVectorLayers.push(casing, center);
    return center;
  }

  function isImpactedRoadMode(layer) {
    return layer === "impact" && ($("impactMapLayer")?.value || "") === "roads";
  }

  function impactFloodSpec(source, date) {
    const byDate = bundle.map?.impact?.[source]?.[date];
    if (byDate?.flood_area) return byDate.flood_area;
    // Backward-compatible fallback for older bundles.
    return bundle.map?.impact?.[source]?.flood_area || bundle.map?.flood?.[source]?.[date]?.extent || null;
  }

  function createMap(id) {
    const map = L.map(id, { zoomControl: true, preferCanvas: true });
    if (mapViewState) {
      map.setView(mapViewState.center, mapViewState.zoom, { animate: false });
    } else {
      map.setView([35.3375, -77.9975], 11, { animate: false });
    }
    map.createPane("rasterPane");
    map.getPane("rasterPane").style.zIndex = 350;
    map.createPane("contextPane");
    map.getPane("contextPane").style.zIndex = 520;
    map.createPane("impactVectorPane");
    map.getPane("impactVectorPane").style.zIndex = 640;

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    addContextLayers(map);
    map.on("moveend zoomend", () => {
      if (mapSyncGuard) return;
      mapViewState = { center: map.getCenter(), zoom: map.getZoom() };
    });
    return map;
  }

  function addContextLayers(map) {
    map._contextLayers = map._contextLayers || {};
    if (contextGeo.upstream_watershed) {
      map._contextLayers.watershed = L.geoJSON(contextGeo.upstream_watershed, {
        pane: "contextPane",
        style: {
          color: "#005a43",
          weight: 4.0,
          opacity: 1.0,
          fillColor: "#52a97d",
          fillOpacity: 0.045,
          dashArray: "10 5",
          lineCap: "round",
          lineJoin: "round",
        },
        onEachFeature: (_f, layer) => layer.bindTooltip("USGS 02089000 upstream watershed", { sticky: true }),
      });
    }
    if (contextGeo.aoi) {
      map._contextLayers.aoi = L.geoJSON(contextGeo.aoi, {
        pane: "contextPane",
        style: { color: "#133f68", weight: 2, fillOpacity: 0, dashArray: "7 5" },
      });
    }
    if (contextGeo.mainstem) {
      map._contextLayers.mainstem = L.geoJSON(contextGeo.mainstem, {
        pane: "contextPane",
        style: { color: "#0b5fa5", weight: 2.5, opacity: 0.9 },
      });
    }
    if (contextGeo.gauge) {
      map._contextLayers.gauge = L.geoJSON(contextGeo.gauge, {
        pane: "contextPane",
        pointToLayer: (_f, latlng) =>
          L.circleMarker(latlng, {
            pane: "contextPane",
            radius: 6,
            color: "#ffffff",
            weight: 2,
            fillColor: "#b01d2e",
            fillOpacity: 1,
          }),
        onEachFeature: (_f, layer) => layer.bindTooltip("USGS 02089000", { direction: "top" }),
      });
    }
    applyContextVisibilityToMap(map);
  }

  function contextWanted(key) {
    const ids = {
      watershed: "toggleWatershed",
      aoi: "toggleAoi",
      mainstem: "toggleMainstem",
      gauge: "toggleGauge",
    };
    const el = $(ids[key]);
    return el ? el.checked : true;
  }

  function applyContextVisibilityToMap(map) {
    if (!map?._contextLayers) return;
    Object.entries(map._contextLayers).forEach(([key, layer]) => {
      const wanted = contextWanted(key);
      const has = map.hasLayer(layer);
      if (wanted && !has) layer.addTo(map);
      if (!wanted && has) map.removeLayer(layer);
    });
  }

  function applyContextVisibility() {
    [singleMap, leftMap, rightMap].forEach(applyContextVisibilityToMap);
  }

  function visibleMap() {
    const dualVisible = !$("dualMaps").classList.contains("hidden");
    if (dualVisible && leftMap) return leftMap;
    return singleMap || leftMap || rightMap;
  }

  function captureVisibleMapView() {
    const map = visibleMap();
    if (!map) return;
    mapViewState = { center: map.getCenter(), zoom: map.getZoom() };
  }

  function restoreMapView(map) {
    if (!map || !mapViewState) return;
    map.setView(mapViewState.center, mapViewState.zoom, { animate: false });
  }

  function fitUpstreamWatershed() {
    const map = visibleMap();
    const layer = map?._contextLayers?.watershed;
    if (!map || !layer) {
      mapMessage("Upstream watershed geometry is not available in this bundle.");
      return;
    }
    const bounds = layer.getBounds();
    if (!bounds?.isValid()) return;
    map.fitBounds(bounds, { padding: [18, 18] });
    mapViewState = { center: map.getCenter(), zoom: map.getZoom() };
    if (leftMap && rightMap && map === leftMap) syncMaps(leftMap, rightMap);
  }

  function clearLayers(map, layers) {
    layers.forEach((layer) => {
      try { map.removeLayer(layer); } catch (_) {}
    });
    layers.length = 0;
  }

  function overlaySpec(layer, source, date) {
    if (layer === "extent" || layer === "depth") {
      return bundle.map?.flood?.[source]?.[date]?.[layer] || null;
    }
    if (layer === "rainfall") return bundle.map?.rainfall?.[date] || null;
    if (layer === "hand") return bundle.map?.hand_equivalent || null;
    if (layer === "wse") return bundle.map?.wse?.[source]?.[date] || null;
    if (layer === "impact") {
      const impactKey = $("impactMapLayer")?.value || "flood_area";
      const byDate = bundle.map?.impact?.[source]?.[date];
      if (byDate?.[impactKey]) return byDate[impactKey];
      // Backward-compatible fallback for older static/peak bundles.
      return bundle.map?.impact?.[source]?.[impactKey] || null;
    }
    return null;
  }

  function addImageOverlay(map, spec, opacityScale = 1, initialOpacity = null) {
    if (!spec) return null;
    const cacheToken = encodeURIComponent(bundle?.build_revision || bundle?.generated_at || bundle?.build || "v2");
    const src = spec.path.includes("?") ? `${spec.path}&v=${cacheToken}` : `${spec.path}?v=${cacheToken}`;
    const layer = L.imageOverlay(src, spec.bounds, {
      opacity: initialOpacity === null ? Math.max(0, Math.min(1, currentOverlayOpacity() * opacityScale)) : initialOpacity,
      interactive: false,
      pane: "rasterPane",
      crossOrigin: false,
    }).addTo(map);
    layer._dashboardOpacityScale = opacityScale;
    return layer;
  }

  function replaceRasterLayer(map, layers, spec, opacityScale = 1, animate = false) {
    if (!map) return null;
    const oldLayers = layers.slice();
    if (!spec) {
      clearLayers(map, layers);
      return null;
    }
    if (!animate || !oldLayers.length) {
      clearLayers(map, layers);
      const fresh = addImageOverlay(map, spec, opacityScale);
      if (fresh) layers.push(fresh);
      return fresh;
    }

    const incoming = addImageOverlay(map, spec, opacityScale, 0);
    if (!incoming) return null;
    layers.push(incoming);
    const started = performance.now();
    const oldStart = oldLayers.map((layer) => {
      try { return Number(layer.options?.opacity ?? layerTargetOpacity(layer)); } catch (_) { return layerTargetOpacity(layer); }
    });
    const newTarget = layerTargetOpacity(incoming);

    const step = (now) => {
      const raw = Math.max(0, Math.min(1, (now - started) / PLAY_FADE_MS));
      // Smoothstep easing avoids the harsh linear flash between daily rasters.
      const t = raw * raw * (3 - 2 * raw);
      try { incoming.setOpacity(newTarget * t); } catch (_) {}
      oldLayers.forEach((layer, idx) => {
        try { layer.setOpacity(oldStart[idx] * (1 - t)); } catch (_) {}
      });
      if (raw < 1) {
        queueFrame();
        return;
      }
      oldLayers.forEach((layer) => {
        try { map.removeLayer(layer); } catch (_) {}
      });
      layers.splice(0, layers.length, incoming);
    };
    const queueFrame = () => {
      const id = window.requestAnimationFrame((now) => {
        playFadeFrames.delete(id);
        step(now);
      });
      playFadeFrames.add(id);
    };
    queueFrame();
    return incoming;
  }

  function fitMapToSpecs(map, specs) {
    const valid = specs.filter(Boolean);
    if (!valid.length) return;
    const bounds = valid.reduce((acc, spec) => {
      const b = L.latLngBounds(spec.bounds);
      return acc ? acc.extend(b) : b;
    }, null);
    if (bounds?.isValid()) map.fitBounds(bounds, { padding: [12, 12], maxZoom: 13 });
  }

  function mapMessage(text) {
    const el = $("mapMessage");
    if (!text) {
      el.classList.add("hidden");
      el.textContent = "";
    } else {
      el.textContent = text;
      el.classList.remove("hidden");
    }
  }

  function renderLandcoverClassLegend(layer) {
    const box = $("landcoverClassLegend");
    const list = $("landcoverClassList");
    if (!box || !list) return;
    const impactKey = $("impactMapLayer")?.value || "";
    const visible = layer === "impact" && impactKey === "landcover";
    box.classList.toggle("hidden", !visible);
    if (!visible) {
      box.open = false;
      return;
    }
    const classes = bundle.impact?.landcover_map_legend || [];
    list.innerHTML = classes.map((item) => `
      <div class="landcover-class-row">
        <span class="landcover-class-swatch" style="background:${item.color}"></span>
        <span class="landcover-class-code">${item.code}</span>
        <span class="landcover-class-name">${item.name}</span>
      </div>
    `).join("");
  }

  function impactRowForSource(source) {
    return (bundle.impact?.rows || []).find((row) => row.source === source) || null;
  }

  function roadLengthChip(source, date) {
    const value = impactedRoadLengthKm(source, date);
    const sourceColor = SOURCE_COLORS[source] || "#51606f";
    const available = hasImpactedRoadVector(source, date);
    const valueText = `${fmt(value, 2)} km`;
    const qualifier = available ? date : `No impacted roads · ${date}`;

    return `
      <span class="road-length-chip">
        <span class="road-source-dot" style="background:${sourceColor}"></span>
        <strong>${sourceLabel(source)}</strong>
        <span>${valueText}</span>
        <small>${qualifier}</small>
      </span>`;
  }

  function renderMapLegend(spec, layer) {
    const el = $("mapLegend");
    renderLandcoverClassLegend(layer);
    if (!spec) {
      el.innerHTML = "";
      return;
    }
    if (layer === "extent" || (layer === "impact" && (($("impactMapLayer")?.value || "") === "flood_area"))) {
      el.innerHTML = `<span class="legend-swatch"></span><strong>Flooded</strong>`;
      return;
    }
    if (layer === "impact") {
      const key = $("impactMapLayer")?.value || "";
      if (key === "roads") {
        const primary = $("primarySource")?.value;
        const secondary = $("secondarySource")?.value;
        const mode = $("compareMode")?.value || "single";
        const sources = mode === "single" ? [primary] : [primary, secondary];
        const uniqueSources = [...new Set(sources.filter(Boolean))];
        el.innerHTML = `
          <strong>Impacted roads</strong>
          <span class="road-length-list">${uniqueSources.map((source) => roadLengthChip(source, $("dateSelect")?.value)).join("")}</span>
          <span class="legend-note">Road lines are clipped to the selected source's flood extent for the selected date. The displayed road length is the selected-date impacted-road length.</span>`;
        return;
      }
      const assetLabels = {
        buildings: "Damaged-building pointers + impacted footprint",
        critical: "Critical-facility damage pointers + impacted footprint",
      };
      if (assetLabels[key]) {
        el.innerHTML = `<strong>${assetLabels[key]}</strong>`;
        return;
      }
      if (key === "landcover") {
        el.innerHTML = `<strong>Impacted land cover</strong><span>NLCD classes · use the class legend at right</span>`;
        return;
      }
    }
    const min = spec.value_min ?? null;
    const max = spec.value_max ?? null;
    const ranges = min !== null && max !== null ? `${fmt(min, 1)} – ${fmt(max, 1)} ${spec.unit || ""}` : (spec.unit || "");
    let gradient = "linear-gradient(90deg,#2c7bb6,#ffffbf,#d7191c)";
    if (layer === "depth") gradient = "linear-gradient(90deg,#deebf7,#6baed6,#08519c)";
    if (layer === "wse") gradient = "linear-gradient(90deg,#440154,#21918c,#fde725)";
    if (layer === "hand") gradient = "linear-gradient(90deg,#2b83ba,#abdda4,#fdae61,#d7191c)";
    if (layer === "impact") {
      const key = $("impactMapLayer")?.value || "";
      if (key === "population") gradient = "linear-gradient(90deg,#ffffe5,#78c679,#006837)";
      if (key === "impervious") gradient = "linear-gradient(90deg,#fff5eb,#fdae6b,#a63603)";
      if (key === "buildings") gradient = "linear-gradient(90deg,#cf433d,#cf433d)";
      if (key === "critical") gradient = "linear-gradient(90deg,#9c27b0,#9c27b0)";
    }
    el.innerHTML = `<strong>${spec.label}</strong><span class="legend-gradient" style="background:${gradient}"></span><span>${ranges}</span>`;
  }

  function availabilityText(layer, date, primary, secondary, mode) {
    const p = overlaySpec(layer, primary, date);
    const s = mode === "single" ? true : overlaySpec(layer, secondary, date);
    if (!p && ["extent", "depth"].includes(layer)) {
      return `No retained ${layer === "extent" ? "flood extent" : "flood depth"} raster for ${date}. Available map dates: ${bundle.map_dates.join(", ")}.`;
    }
    if (!p && layer === "rainfall") return `No daily rainfall preview is available for ${date}.`;
    if (!p && layer === "wse") return `No spatial WSE preview is available for ${sourceLabel(primary)} on ${date}.`;
    if (!p && layer === "hand") return "HAND-equivalent terrain preview is unavailable in this bundle.";
    if (!p && layer === "impact") return `No retained impact-map raster is available for ${sourceLabel(primary)} and the selected impact layer.`;
    if (!s) return `No comparison layer is available for ${sourceLabel(secondary)} on ${date}.`;
    return "";
  }

  function syncMaps(source, target) {
    if (mapSyncGuard) return;
    mapSyncGuard = true;
    target.setView(source.getCenter(), source.getZoom(), { animate: false });
    mapSyncGuard = false;
  }

  function ensureDualMapSync() {
    if (!leftMap || !rightMap || leftMap._syncWired) return;
    leftMap.on("moveend zoomend", () => syncMaps(leftMap, rightMap));
    rightMap.on("moveend zoomend", () => syncMaps(rightMap, leftMap));
    leftMap._syncWired = true;
    rightMap._syncWired = true;
  }

  function renderMap() {
    const date = $("dateSelect").value;
    const primary = $("primarySource").value;
    const secondary = $("secondarySource").value;
    const layer = $("layerSelect").value;
    const mode = $("compareMode").value;

    const roadMode = isImpactedRoadMode(layer);
    const pSpec = overlaySpec(layer, primary, date);
    const sSpec = mode === "single" ? null : overlaySpec(layer, secondary, date);
    const msg = availabilityText(layer, date, primary, secondary, mode);
    mapMessage(msg);
    $("mapAvailability").textContent = msg || `${date} · ${layer.replaceAll("_", " ")} · ${mode.replaceAll("_", " ")}`;
    renderMapLegend(pSpec || sSpec, layer);

    // Flood, rainfall, WSE, and Impact Assessment are date-dependent. Impact
    // previews are now explicitly clipped to the selected daily flood extent.
    const animatePlayback = isPlaying && ["extent", "depth", "rainfall", "wse", "impact"].includes(layer);

    if (mode === "side_by_side" && layerSupportsComparison(layer)) {
      $("singleMap").classList.add("hidden");
      $("dualMaps").classList.remove("hidden");
      if (!leftMap) leftMap = createMap("leftMap");
      if (!rightMap) rightMap = createMap("rightMap");
      ensureDualMapSync();
      setTimeout(() => {
        leftMap.invalidateSize();
        rightMap.invalidateSize();
        restoreMapView(leftMap);
        restoreMapView(rightMap);
      }, 0);

      clearImpactVectorLayers(leftMap);
      clearImpactVectorLayers(rightMap);

      if (roadMode) {
        clearLayers(leftMap, leftRasterLayers);
        clearLayers(rightMap, rightRasterLayers);

        const lFlood = addImageOverlay(leftMap, impactFloodSpec(primary, date), 0.32);
        if (lFlood) leftRasterLayers.push(lFlood);
        if (hasImpactedRoadVector(primary, date)) {
          addImpactedRoadVector(leftMap, primary, date);
        }

        const rFlood = addImageOverlay(rightMap, impactFloodSpec(secondary, date), 0.32);
        if (rFlood) rightRasterLayers.push(rFlood);
        if (hasImpactedRoadVector(secondary, date)) {
          addImpactedRoadVector(rightMap, secondary, date);
        }
      } else {
        replaceRasterLayer(leftMap, leftRasterLayers, pSpec, 1.0, animatePlayback);
        replaceRasterLayer(rightMap, rightRasterLayers, sSpec, 1.0, animatePlayback);
      }

      const tagSuffix = layer === "impact"
        ? `${$("impactMapLayer")?.selectedOptions?.[0]?.textContent || "Impact"} · ${date}`
        : date;
      $("leftMapTag").textContent = `${sourceLabel(primary)} · ${tagSuffix}`;
      $("rightMapTag").textContent = `${sourceLabel(secondary)} · ${tagSuffix}`;

      if (!initialMapFitDone && !mapViewState && (pSpec || sSpec)) {
        fitMapToSpecs(leftMap, [pSpec, sSpec]);
        initialMapFitDone = true;
        mapViewState = { center: leftMap.getCenter(), zoom: leftMap.getZoom() };
      }
      setTimeout(() => syncMaps(leftMap, rightMap), 35);
      return;
    }

    $("dualMaps").classList.add("hidden");
    $("singleMap").classList.remove("hidden");
    if (!singleMap) singleMap = createMap("singleMap");
    setTimeout(() => {
      singleMap.invalidateSize();
      restoreMapView(singleMap);
    }, 0);
    clearImpactVectorLayers(singleMap);

    if (roadMode) {
      clearLayers(singleMap, singleRasterLayers);

      // Always use the selected-date flood as context. Never display the old
      // road-symbol raster in impacted-road mode.
      const flood = addImageOverlay(singleMap, impactFloodSpec(primary, date), 0.32);
      if (flood) singleRasterLayers.push(flood);

      if (hasImpactedRoadVector(primary, date)) {
        addImpactedRoadVector(singleMap, primary, date);
      }
    } else {
      replaceRasterLayer(singleMap, singleRasterLayers, pSpec, 1.0, animatePlayback);
    }

    if (!initialMapFitDone && !mapViewState && (pSpec || sSpec)) {
      fitMapToSpecs(singleMap, [pSpec, sSpec]);
      initialMapFitDone = true;
      mapViewState = { center: singleMap.getCenter(), zoom: singleMap.getZoom() };
    } else {
      restoreMapView(singleMap);
    }
  }

  function selectedHydraulicRow() {
    const source = $("primarySource").value;
    const date = $("dateSelect").value;
    return bundle.hydraulics.find((r) => r.source === source && r.date === date) || null;
  }

  function renderHydraulicCards(row) {
    const cards = row
      ? [
          ["Discharge", fmt(row.discharge_m3s, 1), "m³/s", "Discharge used for this flood state"],
          ["Stage", fmt(row.stage_ft, 2), "ft", "Gauge stage used by retained mapper"],
          ["Gauge WSE", fmt(row.wse_navd88_ft, 2), "ft NAVD88", `${fmt(row.wse_navd88_m, 2)} m NAVD88`],
          ["HAND-equivalent threshold", fmt(row.hand_equivalent_threshold_m, 2), "m", "Gauge-relative mainstem threshold"],
          ["Flood area", fmt(row.flood_area_km2, 2), "km²", "Retained 1 m terrain-map area"],
          ["H-Q branch", row.hq_branch ? row.hq_branch[0].toUpperCase() + row.hq_branch.slice(1) : "—", "", "Branch-aware stage translation"],
        ]
      : [
          ["Discharge", "—", "m³/s", "No hydraulic state for selected date"],
          ["Stage", "—", "ft", ""],
          ["Gauge WSE", "—", "ft NAVD88", ""],
          ["HAND-equivalent threshold", "—", "m", ""],
          ["Flood area", "—", "km²", ""],
          ["H-Q branch", "—", "", ""],
        ];
    $("hydraulicCards").innerHTML = cards
      .map(([label, value, unit, sub]) => `
        <div class="metric-card">
          <div class="label">${label}</div>
          <div class="value">${value}<span class="unit">${unit}</span></div>
          <div class="sub">${sub}</div>
        </div>`)
      .join("");
  }

  function renderHQChart(row) {
    if (hqChart) hqChart.destroy();
    const rising = (bundle.hq_surrogate?.rising || []).map((p) => ({ x: p.q_m3s, y: p.stage_ft }));
    const falling = (bundle.hq_surrogate?.falling || []).map((p) => ({ x: p.q_m3s, y: p.stage_ft }));
    const selected = row ? [{ x: row.discharge_m3s, y: row.stage_ft }] : [];
    hqChart = new Chart($("hqChart"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "Rising branch", data: rising, showLine: true, pointRadius: 0, borderWidth: 2, borderColor: "#2f80ed", backgroundColor: "#2f80ed" },
          { label: "Falling branch", data: falling, showLine: true, pointRadius: 0, borderWidth: 2, borderDash: [6, 4], borderColor: "#7d5aa6", backgroundColor: "#7d5aa6" },
          { label: "Selected state", data: selected, pointRadius: 6, pointHoverRadius: 7, borderWidth: 2, borderColor: "#ffffff", backgroundColor: row ? SOURCE_COLORS[row.source] : "#111" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { title: { display: true, text: "Discharge (m³/s)" }, grid: { color: "rgba(80,100,120,.08)" } },
          y: { title: { display: true, text: "Stage (ft)" }, grid: { color: "rgba(80,100,120,.08)" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  function renderHydraulicDailyChart(source) {
    if (hydraulicChart) hydraulicChart.destroy();
    const rows = bundle.hydraulics.filter((r) => r.source === source).sort((a, b) => a.date.localeCompare(b.date));
    const labels = rows.map((r) => r.date.slice(5));
    hydraulicChart = new Chart($("hydraulicChart"), {
      data: {
        labels,
        datasets: [
          {
            type: "bar",
            label: "Discharge (m³/s)",
            data: rows.map((r) => r.discharge_m3s),
            yAxisID: "yQ",
            backgroundColor: SOURCE_COLORS[source] + "55",
            borderColor: SOURCE_COLORS[source],
            borderWidth: 1,
          },
          {
            type: "line",
            label: "Stage (ft)",
            data: rows.map((r) => r.stage_ft),
            yAxisID: "yFt",
            borderColor: "#7d5aa6",
            backgroundColor: "#7d5aa6",
            pointRadius: 2.5,
            tension: 0.18,
          },
          {
            type: "line",
            label: "WSE NAVD88 (ft)",
            data: rows.map((r) => r.wse_navd88_ft),
            yAxisID: "yFt",
            borderColor: "#159a9c",
            backgroundColor: "#159a9c",
            pointRadius: 2.5,
            borderDash: [5, 3],
            tension: 0.18,
          },
          {
            type: "line",
            label: "HAND-equivalent threshold (m)",
            data: rows.map((r) => r.hand_equivalent_threshold_m),
            yAxisID: "yHand",
            borderColor: "#b06a2b",
            backgroundColor: "#b06a2b",
            pointRadius: 2.5,
            tension: 0.18,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { grid: { display: false }, title: { display: true, text: "September 2018" } },
          yQ: { position: "left", title: { display: true, text: "Discharge (m³/s)" }, grid: { color: "rgba(80,100,120,.07)" } },
          yFt: { position: "right", title: { display: true, text: "Stage / WSE (ft)" }, grid: { drawOnChartArea: false } },
          yHand: { position: "right", offset: true, title: { display: true, text: "HAND threshold (m)" }, grid: { drawOnChartArea: false }, ticks: { maxTicksLimit: 5 } },
        },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      },
    });
  }

  function updateHydraulics() {
    if (!bundle) return;
    const row = selectedHydraulicRow();
    const source = $("primarySource").value;
    const date = $("dateSelect").value;
    $("hydraulicContext").textContent = `${sourceLabel(source)} · ${date}`;
    renderHydraulicCards(row);
    renderHQChart(row);
    renderHydraulicDailyChart(source);
  }

  function renderRainfallChart() {
    if (rainfallChart) rainfallChart.destroy();
    const rows = [...(bundle.rainfall || [])].sort((a, b) => a.date.localeCompare(b.date));
    rainfallChart = new Chart($("rainfallChart"), {
      type: "line",
      data: {
        labels: rows.map((r) => r.date.slice(5)),
        datasets: [
          { label: "AOI mean rainfall (mm)", data: rows.map((r) => r.mean_mm), borderColor: "#2f80ed", backgroundColor: "#2f80ed", tension: .2, pointRadius: 3 },
          { label: "AOI maximum rainfall (mm)", data: rows.map((r) => r.max_mm), borderColor: "#0b5fa5", backgroundColor: "#0b5fa5", borderDash: [6,4], tension: .2, pointRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: { title: { display: true, text: "September 2018" }, grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: "24-hour rainfall (mm)" }, grid: { color: "rgba(80,100,120,.08)" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  function updateRainfallSummary() {
    if (!bundle) return;
    const d = $("dateSelect").value;
    const row = bundle.rainfall.find((r) => r.date === d);
    if (!row) {
      $("rainfallSummary").innerHTML = `
        <div class="rain-stat"><span>Selected date</span><strong>${d}</strong></div>
        <div class="rain-stat"><span>AOI mean</span><strong>—</strong></div>
        <div class="rain-stat"><span>Status</span><strong>No preview</strong></div>`;
      return;
    }
    let status = "Rainfall present";
    if (row.display_status === "dry") status = "Dry / zero field";
    if (row.display_status === "no_valid_data") status = "No valid data";
    $("rainfallSummary").innerHTML = `
      <div class="rain-stat"><span>Selected date</span><strong>${row.date}</strong></div>
      <div class="rain-stat"><span>AOI mean / max</span><strong>${fmt(row.mean_mm, 1)} / ${fmt(row.max_mm, 1)} mm</strong></div>
      <div class="rain-stat"><span>Status</span><strong>${status}</strong></div>`;
  }

  function impactRowsOrdered() {
    const rows = bundle.impact?.rows || [];
    return SOURCE_ORDER.map((s) => rows.find((r) => r.source === s)).filter(Boolean);
  }

  function metricConfig(key) {
    return (bundle.impact?.metrics || []).find((m) => m.key === key) || null;
  }

  function renderImpactMiniCharts() {
    miniCharts.forEach((c) => c.destroy());
    miniCharts = [];
    const grid = $("impactMiniGrid");
    grid.innerHTML = "";
    const rows = impactRowsOrdered();
    (bundle.impact?.metrics || []).forEach((cfg, idx) => {
      const card = document.createElement("article");
      card.className = "mini-chart-card";
      const canvasId = `impactMini${idx}`;
      card.innerHTML = `<h4>${cfg.label}</h4><div class="mini-chart-wrap"><canvas id="${canvasId}"></canvas></div>`;
      grid.appendChild(card);
      const chart = new Chart($(canvasId), {
        type: "bar",
        data: {
          labels: rows.map((r) => r.source_short),
          datasets: [{
            data: rows.map((r) => r[cfg.key]),
            backgroundColor: rows.map((r) => SOURCE_COLORS[r.source]),
            borderRadius: 5,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { beginAtZero: true, ticks: { maxTicksLimit: 5 }, grid: { color: "rgba(80,100,120,.07)" } },
            x: { grid: { display: false } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (ctx) => `${fmt(ctx.raw, cfg.decimals ?? 2)} ${cfg.unit}` } },
          },
        },
      });
      miniCharts.push(chart);
    });
  }

  function renderLandcoverChart() {
    if (landcoverChart) landcoverChart.destroy();
    const categories = bundle.impact?.landcover_categories || [];
    const rows = bundle.impact?.landcover || [];
    const datasets = SOURCE_ORDER
      .filter((s) => rows.some((r) => r.source === s))
      .map((source) => ({
        label: sourceLabel(source),
        data: categories.map((cat) => rows.find((r) => r.source === source && r.category === cat)?.area_km2 ?? null),
        backgroundColor: SOURCE_COLORS[source],
        borderColor: SOURCE_COLORS[source],
        borderWidth: 1,
        borderRadius: 4,
      }));
    landcoverChart = new Chart($("landcoverChart"), {
      type: "bar",
      data: { labels: categories, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { beginAtZero: true, title: { display: true, text: "Affected area (km²)" }, grid: { color: "rgba(80,100,120,.08)" } },
          x: { grid: { display: false }, ticks: { maxRotation: 20, minRotation: 0 } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  function renderImpactSection() {
    $("impactDomain").textContent = bundle.impact?.domain_label || "Modeling AOI";
    const rows = bundle.impact?.rows || [];
    if (!rows.length) {
      $("impactMiniGrid").innerHTML = `<p class="muted">Modeling-AOI 3-source impact data was not found when the bundle was built.</p>`;
      return;
    }
    renderImpactMiniCharts();
    renderLandcoverChart();
  }

  function renderMethodology() {
    $("methodologyGrid").innerHTML = (bundle.methodology || [])
      .map((m) => `<article class="method-card"><h4>${m.title}</h4><p>${m.text}</p></article>`)
      .join("");
    const refCount = bundle.noaa_fim_reference?.rows?.length || 0;
    $("noaaNote").textContent = `NOAA FIM is retained only as a limited-domain hydraulic planning/reference library. ${refCount ? `${refCount} peak-reference source records are present in this bundle.` : "No NOAA peak-reference summary was included in this bundle."} It is not used to extend or constrain the retained modeling-AOI Physics/ML flood maps.`;
  }

  function setBuildStatus() {
    const warnings = bundle.warnings || [];
    const rain = bundle.rainfall_dates?.length || 0;
    const impactMaps = Object.values(bundle.map?.impact || {}).reduce((total, sourceDates) => {
      return total + Object.values(sourceDates || {}).reduce((n, dayMap) => n + Object.keys(dayMap || {}).length, 0);
    }, 0);
    $("buildStatus").textContent = warnings.length
      ? `Ready · ${warnings.length} warning${warnings.length === 1 ? "" : "s"}`
      : `Ready · ${rain} rainfall days · ${impactMaps} impact-map previews`;
  }

  async function init() {
    try {
      const response = await fetch("../data_bundle.json", { cache: "no-store" });
      if (!response.ok) throw new Error(`data_bundle.json: HTTP ${response.status}`);
      bundle = await response.json();
      setBuildStatus();
      await loadContextGeo();
      await loadImpactVectors();
      hydrateControls();
      renderMap();
      updateHydraulics();
      renderRainfallChart();
      updateRainfallSummary();
      renderImpactSection();
      renderMethodology();
      activateMapView();
      updatePlayUi();
    } catch (error) {
      console.error(error);
      $("buildStatus").textContent = "Dashboard failed to load";
      mapMessage(`Dashboard data could not be loaded: ${error.message}. Run build_dashboard_v2.py and serve output/frontend_v2 over HTTP.`);
    }
  }

  window.addEventListener("beforeunload", () => stopDatePlayback(false));
  window.addEventListener("DOMContentLoaded", init);
})();
