"""
Topology-aware retrieval (Phase 4).

Incident (from detection) + BuildingKnowledgeGraph -> EvidencePacket:
the alarmed assets, their common supply point, the supply path back to
the source, the physical neighbourhood, and the sensors that stayed
normal -- serialised with graph IDs so Phase 5 can check every claim.

    SubgraphRetriever  -- builds the packet
    RetrievalConfig    -- seeds cap, BFS hops, negative-evidence cap
    EvidencePacket     -- the serialisable result, with a leakage guard
"""

from .evidence import EvidencePacket
from .retriever import RetrievalConfig, SubgraphRetriever

__all__ = ["EvidencePacket", "RetrievalConfig", "SubgraphRetriever"]
