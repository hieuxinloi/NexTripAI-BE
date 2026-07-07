from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

TState = TypeVar("TState")


class BaseAgentGraph(ABC, Generic[TState]):
    graph_name: str

    @property
    @abstractmethod
    def state_schema(self) -> type[TState]:
        raise NotImplementedError

    @abstractmethod
    def register_nodes(self, graph: StateGraph) -> StateGraph:
        raise NotImplementedError

    @abstractmethod
    def register_edges(self, graph: StateGraph) -> StateGraph:
        raise NotImplementedError

    def build(self, *, checkpointer: object | None = None) -> CompiledStateGraph:
        graph = StateGraph(self.state_schema)
        graph = self.register_nodes(graph)
        graph = self.register_edges(graph)
        return graph.compile(checkpointer=checkpointer)
