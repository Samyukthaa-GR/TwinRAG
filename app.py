import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(
    page_title="TwinRAG Demo",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.metric-label {
    font-size: 14px;
    color: #666;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
}
.metric-card {
    background: var(--secondary-background-color);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.2);
}

.metric-label {
    color: var(--text-color);
    opacity: 0.75;
    font-size: 14px;
}

.metric-value {
    color: var(--text-color);
    font-size: 24px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">TwinRAG Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Digital Twin-Driven GraphRAG for Hallucination-Free Root Cause Analysis</div>',
    unsafe_allow_html=True
)

if "fault_injected" not in st.session_state:
    st.session_state.fault_injected = False

btn1, btn2 = st.columns([1, 5])

with btn1:
    if st.button("Inject Leak"):
        st.session_state.fault_injected = True

with btn2:
    if st.button("Reset System"):
        st.session_state.fault_injected = False

status = "Anomaly Detected" if st.session_state.fault_injected else "Normal"
fault_asset = "Pipe P-2" if st.session_state.fault_injected else "None"
pressure_drop = "34%" if st.session_state.fault_injected else "0%"
confidence = "96%" if st.session_state.fault_injected else "--"

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">System Status</div>
        <div class="metric-value">{status}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Faulty Asset</div>
        <div class="metric-value">{fault_asset}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Pressure Drop</div>
        <div class="metric-value">{pressure_drop}</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Diagnosis Confidence</div>
        <div class="metric-value">{confidence}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

G = nx.Graph()

nodes = {
    "Pump A": (0, 1),
    "Pipe P-1": (1, 1),
    "Valve V-1": (2, 1),
    "Junction J-1": (3, 1),
    "Pipe P-2": (4, 1),
    "Junction J-2": (5, 1),
    "Tank T-1": (2, 2),
    "Sensor S-1": (4, 2),
}

edges = [
    ("Pump A", "Pipe P-1"),
    ("Pipe P-1", "Valve V-1"),
    ("Valve V-1", "Junction J-1"),
    ("Junction J-1", "Pipe P-2"),
    ("Pipe P-2", "Junction J-2"),
    ("Tank T-1", "Valve V-1"),
    ("Sensor S-1", "Pipe P-2"),
]

G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

def get_node_color(node):
    if st.session_state.fault_injected and node == "Pipe P-2":
        return "#e74c3c"
    if st.session_state.fault_injected and node in ["Junction J-1", "Junction J-2", "Sensor S-1"]:
        return "#f39c12"
    if "Pump" in node:
        return "#3498db"
    if "Valve" in node:
        return "#9b59b6"
    if "Tank" in node:
        return "#1abc9c"
    if "Sensor" in node:
        return "#34495e"
    return "#85c1e9"

edge_x, edge_y = [], []

for edge in G.edges():
    x0, y0 = nodes[edge[0]]
    x1, y1 = nodes[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, node_colors = [], [], []

for node in G.nodes():
    x, y = nodes[node]
    node_x.append(x)
    node_y.append(y)
    node_colors.append(get_node_color(node))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    line=dict(width=4, color="#a8bdbd"),
    hoverinfo="none"
))

fig.add_trace(go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=list(G.nodes()),
    textposition="bottom center",
    marker=dict(
        size=34,
        color=node_colors,
        line=dict(width=2, color="white")
    ),
    hoverinfo="text"
))

fig.update_layout(
    height=460,
    showlegend=False,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="black",
    paper_bgcolor="white",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False),
)

left, right = st.columns([1.6, 1])

with left:
    st.subheader("Digital Twin Network View")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("TwinRAG Diagnosis")

    if not st.session_state.fault_injected:
        st.success("System operating normally.")
        st.write(
            """
            The Digital Twin baseline and simulated sensor readings are aligned.
            No significant residual deviation has been detected.
            """
        )
    else:
        st.error("Root Cause Identified")
        st.write(
            """
            A pressure anomaly was detected at **Pipe P-2**.

            The Digital Twin predicted normal flow and pressure values, but the
            simulated sensor stream shows a **34% pressure drop**.

            The topology-aware retrieval layer identified the directly connected
            components: **Junction J-1**, **Pipe P-2**, **Junction J-2**, and
            **Sensor S-1**.

            Based only on this verified graph context, TwinRAG concludes that the
            most probable root cause is a **leak in Pipe P-2**.
            """
        )

st.divider()

bottom1, bottom2 = st.columns(2)

with bottom1:
    st.subheader("Retrieved Graph Context")
    if st.session_state.fault_injected:
        st.code(
            """
Junction J-1  →  Pipe P-2  →  Junction J-2
                     ↑
                 Sensor S-1

Retrieved using topology-aware graph traversal.
            """
        )
    else:
        st.info("Graph context will appear after an anomaly is detected.")

with bottom2:
    st.subheader("Constrained LLM Prompt")
    if st.session_state.fault_injected:
        st.code(
            """
You are a fault diagnosis assistant.

Use only the verified topology below.

Faulty Asset: Pipe P-2
Pressure Drop: 34%

Connected Components:
- Junction J-1
- Pipe P-2
- Junction J-2
- Sensor S-1

Task:
Identify the most probable root cause.
Do not mention components outside the given topology.
            """
        )
    else:
        st.info("LLM prompt will be generated after graph context retrieval.")