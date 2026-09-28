# LangGraph Architecture Visualization

Here is the visual representation of our LangGraph state machine. You can see how the `GraphState` (the shared memory) flows linearly through the 5 nodes we built.

```mermaid
graph TD
    %% Define styles
    classDef node fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef startend fill:#d4002a,stroke:#fff,stroke-width:2px,color:#fff;
    classDef state fill:#e8f4f8,stroke:#00a3e0,stroke-width:2px,stroke-dasharray: 5 5;

    %% Entry point
    START((START)):::startend

    %% Nodes
    N1[understand_query<br/><i>Detects Language</i>]:::node
    N2[retrieve<br/><i>Queries ChromaDB</i>]:::node
    N3[assemble_context<br/><i>Formats Chunks</i>]:::node
    N4[generate<br/><i>Calls Azure OpenAI</i>]:::node
    N5[respond<br/><i>Filters Citations</i>]:::node

    %% Exit point
    END((END)):::startend

    %% State Object (Memory)
    State[\"GraphState"<br/>- user_query<br/>- chat_history<br/>- language<br/>- retrieved_docs<br/>- context<br/>- answer<br/>- sources/]:::state

    %% Connections
    START --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> END

    %% State flow representation
    State -.-> N1
    State -.-> N2
    State -.-> N3
    State -.-> N4
    State -.-> N5
```

### What happens at each step:
1. **START:** The user submits a question and Streamlit passes the `chat_history`.
2. **understand_query:** Updates the state with `language="en"` or `"ar"`.
3. **retrieve:** Reads the `language`, queries the vector DB, and saves `retrieved_docs` to the state.
4. **assemble_context:** Takes `retrieved_docs` and formats them into a single `context` string.
5. **generate:** Sends the `context`, `chat_history`, and `user_query` to the LLM. Saves the `answer` to the state.
6. **respond:** Checks the `answer` for inline citations (e.g., `[1]`) and saves the matched `sources`.
7. **END:** The final state is returned to the Streamlit app to be displayed!
