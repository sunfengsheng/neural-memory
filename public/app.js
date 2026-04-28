// ── Neural Memory Visualizer ────────────────────────────────────────
// D3.js force-directed graph + detail panel, powered by Express REST
// which bridges to neural-memory MCP over stdio.

// ── Color maps ─────────────────────────────────────────────────────
const TYPE_COLOR = {
  semantic: "#4fc3f7",
  episodic: "#ff8a65",
  procedural: "#81c784",
  schema: "#ce93d8",
};
const SYNAPSE_COLOR = {
  semantic: "#90a4ae",
  temporal: "#ffd54f",
  causal: "#ef5350",
  hierarchical: "#7e57c2",
  reference: "#26a69a",
};

// ── State ──────────────────────────────────────────────────────────
let graphData = { neurons: [], synapses: [] };
let simulation = null;
let selectedNeuron = null;
let svg, g, linkG, nodeG, tooltip;
let W, H;

// ── Boot ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const box = document.getElementById("graph-container");
  W = box.clientWidth;
  H = box.clientHeight;

  svg = d3.select("#graph").attr("viewBox", [0, 0, W, H]);

  const zoom = d3
    .zoom()
    .scaleExtent([0.1, 8])
    .on("zoom", (e) => g.attr("transform", e.transform));
  svg.call(zoom);

  // Arrow markers per synapse type
  const defs = svg.append("defs");
  for (const [type, color] of Object.entries(SYNAPSE_COLOR)) {
    defs
      .append("marker")
      .attr("id", `arrow-${type}`)
      .attr("viewBox", "0 -4 8 8")
      .attr("refX", 20)
      .attr("markerWidth", 5)
      .attr("markerHeight", 5)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-4L8,0L0,4")
      .attr("fill", color);
  }

  g = svg.append("g");
  linkG = g.append("g");
  nodeG = g.append("g");

  tooltip = d3
    .select("body")
    .append("div")
    .attr("class", "tooltip")
    .style("display", "none");

  // Buttons
  document.getElementById("btn-refresh").onclick = loadGraph;
  document.getElementById("btn-reflect").onclick = doReflect;
  document.getElementById("btn-remember").onclick = showRememberModal;
  document.getElementById("btn-delete").onclick = doDelete;
  document.getElementById("modal-cancel").onclick = hideModal;
  document.getElementById("modal-ok").onclick = () => {
    if (window._modalOk) window._modalOk();
  };

  // Deselect on background click
  svg.on("click", () => deselectNode());

  // Resize
  window.addEventListener("resize", () => {
    W = box.clientWidth;
    H = box.clientHeight;
    svg.attr("viewBox", [0, 0, W, H]);
  });

  loadGraph();
});

// ── API ────────────────────────────────────────────────────────────
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── Load data ──────────────────────────────────────────────────────
async function loadGraph() {
  try {
    graphData = await api("/api/graph");
    render();
    loadStatus();
  } catch (e) {
    console.error("loadGraph:", e);
  }
}

async function loadStatus() {
  try {
    const s = await api("/api/status");
    const n = s.neurons || {};
    const sy = s.synapses || {};
    document.getElementById("stats").textContent =
      `Neurons: ${n.total ?? 0}  (W:${n.by_layer?.working ?? 0}  S:${n.by_layer?.short_term ?? 0}  L:${n.by_layer?.long_term ?? 0})` +
      `  |  Synapses: ${sy.total ?? 0}` +
      `  |  Embedding: ${s.embedding_dimension ?? "?"}d`;
  } catch (_) {}
}

// ── Render ──────────────────────────────────────────────────────────
function render() {
  const { neurons, synapses } = graphData;
  const idSet = new Set(neurons.map((n) => n.id));

  const links = synapses
    .filter((s) => idSet.has(s.pre_neuron_id) && idSet.has(s.post_neuron_id))
    .map((s) => ({
      source: s.pre_neuron_id,
      target: s.post_neuron_id,
      type: s.synapse_type,
      weight: s.weight,
      _id: s.id,
    }));

  const nodes = neurons.map((n) => ({ ...n }));

  if (simulation) simulation.stop();

  simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance(100)
        .strength((d) => (d.weight || 0.5) * 0.3)
    )
    .force("charge", d3.forceManyBody().strength(-150))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collide", d3.forceCollide().radius((d) => radius(d) + 4));

  // ── Links ──
  linkG.selectAll("line").remove();
  const linkSel = linkG
    .selectAll("line")
    .data(links, (d) => d._id)
    .join("line")
    .attr("stroke", (d) => SYNAPSE_COLOR[d.type] || "#555")
    .attr("stroke-width", (d) => Math.max(0.8, (d.weight || 0.5) * 3))
    .attr("stroke-opacity", 0.45)
    .attr("marker-end", (d) => `url(#arrow-${d.type})`);

  // ── Nodes ──
  nodeG.selectAll(".node").remove();
  const nodeSel = nodeG
    .selectAll(".node")
    .data(nodes, (d) => d.id)
    .join("g")
    .attr("class", "node")
    .call(drag(simulation));

  // Draw shape
  nodeSel.each(function (d) {
    const el = d3.select(this);
    const r = radius(d);
    const color = TYPE_COLOR[d.neuron_type] || "#999";

    if (d.layer === "short_term") {
      // Diamond
      el.append("rect")
        .attr("width", r * 1.6)
        .attr("height", r * 1.6)
        .attr("x", (-r * 1.6) / 2)
        .attr("y", (-r * 1.6) / 2)
        .attr("transform", "rotate(45)")
        .attr("fill", color)
        .attr("fill-opacity", 0.65)
        .attr("stroke", color)
        .attr("stroke-width", 1.5);
    } else if (d.layer === "long_term") {
      // Square
      el.append("rect")
        .attr("width", r * 2)
        .attr("height", r * 2)
        .attr("x", -r)
        .attr("y", -r)
        .attr("rx", 3)
        .attr("fill", color)
        .attr("fill-opacity", 0.65)
        .attr("stroke", color)
        .attr("stroke-width", 1.5);
    } else {
      // Circle (working)
      el.append("circle")
        .attr("r", r)
        .attr("fill", color)
        .attr("fill-opacity", 0.65)
        .attr("stroke", color)
        .attr("stroke-width", 1.5);
    }

    // Strength ring (outer glow)
    if (d.layer !== "short_term") {
      el.append("circle")
        .attr("r", r + 3)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 0.5)
        .attr("stroke-opacity", (d.strength || 0) * 0.6);
    }

    // Label
    el.append("text")
      .attr("dy", r + 14)
      .attr("text-anchor", "middle")
      .attr("fill", "#8899aa")
      .attr("font-size", "9px")
      .text(truncate(d.summary || d.content, 18));
  });

  // Hover / click
  nodeSel
    .on("mouseover", (ev, d) => {
      tooltip
        .style("display", "block")
        .html(
          `<b style="color:${TYPE_COLOR[d.neuron_type] || "#ccc"}">${d.neuron_type}</b> [${d.layer}]<br>` +
            truncate(d.content, 140) +
            `<br><span style="color:#667">str:${f(d.strength)} imp:${f(d.importance)} acc:${d.access_count || 0}</span>`
        );
    })
    .on("mousemove", (ev) => {
      tooltip
        .style("left", ev.pageX + 14 + "px")
        .style("top", ev.pageY - 8 + "px");
    })
    .on("mouseout", () => tooltip.style("display", "none"))
    .on("click", (ev, d) => {
      ev.stopPropagation();
      selectNode(d, nodeSel, linkSel);
    });

  // Tick
  simulation.on("tick", () => {
    linkSel
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);
    nodeSel.attr("transform", (d) => `translate(${d.x},${d.y})`);
  });
}

function radius(d) {
  return 6 + (d.importance || 0.5) * 8 + (d.strength || 0.5) * 4;
}

function drag(sim) {
  return d3
    .drag()
    .on("start", (e, d) => {
      if (!e.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (e, d) => {
      d.fx = e.x;
      d.fy = e.y;
    })
    .on("end", (e, d) => {
      if (!e.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    });
}

// ── Selection ──────────────────────────────────────────────────────
function selectNode(d, nodeSel, linkSel) {
  selectedNeuron = d;
  document.getElementById("panel-placeholder").classList.add("hidden");
  document.getElementById("panel-content").classList.remove("hidden");
  document.getElementById("detail-title").textContent =
    `${d.neuron_type.toUpperCase()} [${d.layer}]`;

  const tags = (d.tags || [])
    .map((t) => `<span class="tag">${esc(t)}</span>`)
    .join(" ");

  document.getElementById("detail-body").innerHTML = `
    <div class="field">
      <div class="label">Content</div>
      <div class="content-preview">${esc(d.content)}</div>
    </div>
    <div class="field">
      <div class="label">ID</div>
      <div class="value" style="font-size:10px;color:#556;word-break:break-all">${d.id}</div>
    </div>
    <div class="field">
      <div class="label">Strength / Stability / Importance</div>
      <div class="value">${f(d.strength)} / ${f(d.stability)} / ${f(d.importance)}</div>
    </div>
    <div class="field">
      <div class="label">Emotion (valence / arousal)</div>
      <div class="value">${f(d.emotional_valence)} / ${f(d.emotional_arousal)}</div>
    </div>
    <div class="field">
      <div class="label">Access count</div>
      <div class="value">${d.access_count ?? 0}</div>
    </div>
    <div class="field">
      <div class="label">Tags</div>
      <div class="value">${tags || '<span style="color:#556">none</span>'}</div>
    </div>
    <div class="field">
      <div class="label">Source</div>
      <div class="value">${d.source || "?"}</div>
    </div>
    <div class="field">
      <div class="label">Created</div>
      <div class="value">${fmtDate(d.created_at)}</div>
    </div>
    <div class="field">
      <div class="label">Last accessed</div>
      <div class="value">${fmtDate(d.last_accessed)}</div>
    </div>
  `;

  // Highlight connected
  const connectedIds = new Set();
  graphData.synapses.forEach((s) => {
    if (s.pre_neuron_id === d.id) connectedIds.add(s.post_neuron_id);
    if (s.post_neuron_id === d.id) connectedIds.add(s.pre_neuron_id);
  });

  if (nodeSel)
    nodeSel.attr("opacity", (n) =>
      n.id === d.id ? 1 : connectedIds.has(n.id) ? 0.85 : 0.2
    );
  if (linkSel)
    linkSel.attr("stroke-opacity", (l) => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return s === d.id || t === d.id ? 0.9 : 0.05;
    });
}

function deselectNode() {
  selectedNeuron = null;
  document.getElementById("panel-placeholder").classList.remove("hidden");
  document.getElementById("panel-content").classList.add("hidden");
  nodeG.selectAll(".node").attr("opacity", 1);
  linkG.selectAll("line").attr("stroke-opacity", 0.45);
}

// ── Actions ────────────────────────────────────────────────────────
async function doReflect() {
  try {
    const d = await api("/api/reflect", { method: "POST" });
    alert(
      `Reflect done:\n` +
        `  Decayed: ${d.decayed}\n  Pruned: ${d.pruned}\n` +
        `  W->S: ${d.working_to_short_term}\n  S->L: ${d.short_term_to_long_term}\n` +
        `  Schemas: ${d.schemas_created}\n  Syn decayed: ${d.synapse_decayed}`
    );
    loadGraph();
  } catch (e) {
    alert("Reflect error: " + e.message);
  }
}

async function doDelete() {
  if (!selectedNeuron) return;
  if (
    !confirm(
      `Delete this memory?\n\n${truncate(selectedNeuron.content, 80)}`
    )
  )
    return;
  try {
    await api("/api/forget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ neuron_id: selectedNeuron.id, confirm: true }),
    });
    deselectNode();
    loadGraph();
  } catch (e) {
    alert("Delete error: " + e.message);
  }
}

// ── Remember modal ─────────────────────────────────────────────────
function showRememberModal() {
  showModal(
    "Create New Memory",
    `<label>Content</label>
     <textarea id="inp-content" rows="4" placeholder="What to remember..."></textarea>
     <label>Type</label>
     <select id="inp-type">
       <option value="semantic" selected>Semantic</option>
       <option value="episodic">Episodic</option>
       <option value="procedural">Procedural</option>
     </select>
     <label>Importance (0-1)</label>
     <input type="number" id="inp-imp" value="0.5" min="0" max="1" step="0.1"/>
     <label>Tags (comma-separated)</label>
     <input type="text" id="inp-tags" placeholder="tag1, tag2"/>`,
    async () => {
      const content = document.getElementById("inp-content").value.trim();
      if (!content) return alert("Content required");
      const tags = document.getElementById("inp-tags").value.trim();
      try {
        await api("/api/remember", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            content,
            neuron_type: document.getElementById("inp-type").value,
            importance: parseFloat(document.getElementById("inp-imp").value),
            tags: tags
              ? JSON.stringify(tags.split(",").map((t) => t.trim()))
              : "[]",
            source: "web-ui",
          }),
        });
        hideModal();
        loadGraph();
      } catch (e) {
        alert("Remember error: " + e.message);
      }
    }
  );
}

// ── Modal plumbing ─────────────────────────────────────────────────
function showModal(title, html, onOk) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").innerHTML = html;
  document.getElementById("modal-overlay").classList.remove("hidden");
  window._modalOk = onOk;
}
function hideModal() {
  document.getElementById("modal-overlay").classList.add("hidden");
  window._modalOk = null;
}

// ── Util ───────────────────────────────────────────────────────────
function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, n) + "..." : s;
}
function f(v) {
  return v != null ? Number(v).toFixed(2) : "?";
}
function esc(t) {
  const d = document.createElement("div");
  d.textContent = t || "";
  return d.innerHTML;
}
function fmtDate(iso) {
  if (!iso) return "?";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}
