import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Tuple, Union
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="CoRE System",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tomorrow:wght@400;500;600&display=swap');
    
    /* Main theme colors */
    :root {
        --background-color: #1E201E;
        --text-color: #76ABAE;
        --accent-color: #EEEEEE;
        --secondary-color: #31363F;
        --error-color: #CF6679;
    }
    
    /* Override Streamlit's default styles */
    .stApp {
        background-color: var(--background-color);
    }
    
    .stMarkdown, .stText, p, span {
        color: var(--text-color) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-family: 'Tomorrow', sans-serif;
    }
    
    /* Buttons */
    .stButton > button:hover {
        background-color: var(--accent-color);
        color: black;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-family: 'Tomorrow', sans-serif;
    }
    
    .stButton > button {
        background-color: var(--secondary-color);
        color: black;
    }
    
    /* Additional styles remain the same as in original CSS */
    </style>
""", unsafe_allow_html=True)

class GeminiExpert:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini Expert with API configuration
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def process(self, query: str, role: str, context: str = "") -> str:
        """
        Process query using Gemini model with context
        """
        self.role=role
        full_prompt = f"""You are an expert with the role: {role}.
            Previous context: {context}
            Please analyze and respond to the following query: {query}
            Analyse the previous context and iterate upon it with your perspective, correct any info if required.
            Provide your expert perspective while staying focused on your role."""
        
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Gemini Expert Error: {str(e)}"

class KnowledgeTransferHub:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.shared_knowledge_base: List[Dict[str, Any]] = []
    
    def extract_common_insights(self, expert_responses: List[Dict[str, Any]]) -> List[str]:
        """
        Extract common insights across expert responses
        """
        # Check if responses are empty or insufficient
        if not expert_responses or len(expert_responses) < 2:
            return []
        
        texts = [resp['response'] for resp in expert_responses]
        
        # Ensure non-empty texts
        texts = [text for text in texts if text and isinstance(text, str)]
        
        if len(texts) < 2:
            return []
        
        try:
            embeddings = self.embedding_model.encode(texts)
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Compute pairwise similarities
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find clusters of similar responses
            common_insights = []
            for i in range(len(texts)):
                similar_indices = np.where(similarity_matrix[i] > 0.7)[0]
                if len(similar_indices) > 1:
                    insight = texts[i]
                    common_insights.append(insight)
            
            return common_insights
        except Exception as e:
            print(f"Error in extracting common insights: {e}")
            return []
    
    def propagate_knowledge(self, expert_responses: List[Dict[str, Any]]):
        """
        Propagate knowledge across experts and update shared knowledge base
        """
        common_insights = self.extract_common_insights(expert_responses)
        
        for insight in common_insights:
            # Skip empty or invalid insights
            if not insight or not isinstance(insight, str):
                continue
            
            try:
                # Check if insight already exists
                insight_embedding = self.embedding_model.encode([insight])
                
                # Ensure 2D array for existing knowledge base
                if len(self.shared_knowledge_base) > 0:
                    existing_embeddings = self.embedding_model.encode(
                        [existing['text'] for existing in self.shared_knowledge_base]
                    )
                    
                    # Compute similarity, handling potential shape issues
                    if insight_embedding.ndim == 1:
                        insight_embedding = insight_embedding.reshape(1, -1)
                    
                    similarity = cosine_similarity(insight_embedding, existing_embeddings)[0]
                    exists = any(sim > 0.8 for sim in similarity)
                else:
                    exists = False
                
                if not exists:
                    self.shared_knowledge_base.append({
                        'text': insight,
                        'timestamp': len(self.shared_knowledge_base)
                    })
            except Exception as e:
                print(f"Error in propagating knowledge: {e}")
        
        # Limit shared knowledge base size
        if len(self.shared_knowledge_base) > 100:
            self.shared_knowledge_base = self.shared_knowledge_base[-100:]
    
    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant knowledge from shared knowledge base
        """
        # Handle empty or invalid shared knowledge base
        if not self.shared_knowledge_base:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            
            # Ensure 2D array
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            knowledge_embeddings = self.embedding_model.encode(
                [item['text'] for item in self.shared_knowledge_base]
            )
            
            # Ensure 2D array for knowledge embeddings
            if knowledge_embeddings.ndim == 1:
                knowledge_embeddings = knowledge_embeddings.reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
            
            # Handle case with insufficient similarities
            if len(similarities) == 0:
                return []
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            return [self.shared_knowledge_base[i]['text'] for i in top_indices]
        except Exception as e:
            print(f"Error in retrieving relevant knowledge: {e}")
            return []

class ContextManager:
    def __init__(self, max_short_term_length: int = 5, 
                 max_medium_term_length: int = 20, 
                 max_long_term_length: int = 50):
        # Initialize semantic embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Context storage tiers
        self.short_term_context: List[Dict[str, Any]] = []
        self.medium_term_context: List[Dict[str, Any]] = []
        self.long_term_context: List[Dict[str, Any]] = []
        
        # Tier length limits
        self.max_short_term = max_short_term_length
        self.max_medium_term = max_medium_term_length
        self.max_long_term = max_long_term_length
    
    def semantic_compress(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Semantically compress context by removing redundant information
        """
        if not context or len(context) < 2:
            return context
        
        # Extract texts, filtering out invalid entries
        texts = [item['text'] for item in context if item.get('text') and isinstance(item['text'], str)]
        
        if len(texts) < 2:
            return context
        
        try:
            embeddings = self.embedding_model.encode(texts)
            
            # Ensure 2D array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Select unique and most representative contexts
            unique_indices = []
            for i in range(len(texts)):
                is_unique = True
                for j in unique_indices:
                    if similarity_matrix[i][j] > 0.8:  # High similarity threshold
                        is_unique = False
                        break
                if is_unique:
                    unique_indices.append(i)
            
            return [context[texts.index(texts[i])] for i in unique_indices]
        except Exception as e:
            print(f"Error in semantic compression: {e}")
            return context
    
    def add_context(self, model: str, role: str, text: str, tier: str = 'short'):
        """
        Add context to appropriate tier with semantic considerations
        """
        # Skip empty or invalid context
        if not text or not isinstance(text, str):
            return
        
        context_entry = {
            'model': model,
            'role': role,
            'text': text,
            'timestamp': len(self.short_term_context)
        }
        
        # Select appropriate context tier
        if tier == 'short':
            self.short_term_context.append(context_entry)
            if len(self.short_term_context) > self.max_short_term:
                self.short_term_context = self.semantic_compress(self.short_term_context)
                self.short_term_context = self.short_term_context[-self.max_short_term:]
                self.medium_term_context.extend(self.short_term_context)
        
        elif tier == 'medium':
            self.medium_term_context.append(context_entry)
            if len(self.medium_term_context) > self.max_medium_term:
                self.medium_term_context = self.semantic_compress(self.medium_term_context)
                self.medium_term_context = self.medium_term_context[-self.max_medium_term:]
                self.long_term_context.extend(self.medium_term_context)
        
        elif tier == 'long':
            self.long_term_context.append(context_entry)
            if len(self.long_term_context) > self.max_long_term:
                self.long_term_context = self.semantic_compress(self.long_term_context)
                self.long_term_context = self.long_term_context[-self.max_long_term:]
    
    def get_context(self, tier: str = 'short') -> str:
        """
        Retrieve context for a specific tier
        """
        if tier == 'short':
            context = self.short_term_context
        elif tier == 'medium':
            context = self.medium_term_context + self.short_term_context
        else:
            context = self.long_term_context + self.medium_term_context + self.short_term_context
        
        # Filter out invalid context entries
        valid_context = [item for item in context if item.get('text') and isinstance(item['text'], str)]
        
        return '\n'.join([f"{item['model']} ({item['role']}): {item['text']}" for item in valid_context])

class ExpertGating:
    def __init__(self, default_llm: str = "llama3.2:3b"):
        self.llm = Ollama(model=default_llm)
        
    def analyze_query(self, query: str) -> Dict[str, float]:
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Analyze the following query and assign relevance scores (0-1) for different types of expertise needed:
            Query: {query}
            
            Consider these aspects and output only the JSON:
            - Conversational (how much social interaction/dialogue is needed)
            - Logical (how much reasoning/analysis is required)
            - Creative (how much imagination/innovation is needed)
            - Engagement (how much explanation/elaboration is required)"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(query=query)
        
        try:
            scores = json.loads(response)
            return {
                "gemma": scores.get("Conversational", 0.5),
                "llama": scores.get("Logical", 0.5),
                "qwen": scores.get("Creative", 0.5),
                "phi": scores.get("Engagement", 0.5)
            }
        except:
            return {"gemma": 0.5, "llama": 0.5, "qwen": 0.5, "phi": 0.5}

class Expert:
    def __init__(self, model_name: str, role: str = None):
        self.model_name = model_name
        self.role = role
        self.llm = Ollama(model=model_name)
        
    def process(self, query: str, context: str = "") -> str:
        prompt_template = PromptTemplate(
            input_variables=["role", "context", "query"],
            template="""You are an expert with the role: {role}.
            Previous context: {context}
            Please analyze and respond to the following query: {query}
            Analyse the previous context and iterate upon it with your perspective, correct any info if required.
            Provide your expert perspective while staying focused on your role."""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        response = chain.run(role=self.role, context=context, query=query)
        return response

class MoRESystem:
    def __init__(self, gemini_api_key: str = None):
        self.available_models = {
            "gemma": "gemma2:2b",
            "llama": "llama3.2:3b",
            "phi": "phi3:latest",
            "qwen": "qwen2.5:3b",
            "gemini": "gemini-2.0-flash"
        }
        
        self.default_roles = {
            "gemma": "Case Study Expert",
            "llama": "Logical Reasoning Expert",
            "phi": "Engagement Expert",
            "qwen": "Creative Expert",
            "gemini": "Code writing Expert"
        }
        
        self.experts: Dict[str, Union[Expert, GeminiExpert]] = {}
        self.gating = ExpertGating()
        
        # New context and knowledge transfer modules
        self.context_manager = ContextManager()
        self.knowledge_transfer = KnowledgeTransferHub()
        
        # Gemini API key (optional)
        self.gemini_api_key = gemini_api_key
    
    def initialize_experts(self, selected_models: List[str], custom_roles: Dict[str, str] = None):
        """
        Initialize experts with either custom or default roles
        
        Args:
        selected_models (List[str]): List of models to initialize
        custom_roles (Dict[str, str], optional): Dictionary of custom roles
        """
        self.experts = {}
        
        # Ensure custom_roles is a valid dictionary
        if custom_roles is None:
            custom_roles = {}
        
        for model in selected_models:
       
            role = (
                custom_roles.get(model) or  
                self.default_roles.get(model) or 
                f"{model.capitalize()} Expert"  
            )
            
            # Special handling for Gemini
            if model == "gemini":
                if not self.gemini_api_key:
                    raise ValueError("Gemini API key is required for Gemini model")
                self.experts[model] = GeminiExpert(self.gemini_api_key)
                # Explicitly set the role for Gemini
                self.experts[model].role = role
            else:
                # Create Expert with the selected/default role
                self.experts[model] = Expert(self.available_models[model], role)
    
    def process_query(self, query: str, aggregator_model: str) -> Tuple[str, List[Dict]]:
        context = self.context_manager.get_context('medium')  # Get medium-term context
        
        # Retrieve relevant knowledge from shared knowledge base
        relevant_knowledge = self.knowledge_transfer.retrieve_relevant_knowledge(query)
        context += "\n\nRelevant Past Knowledge:\n" + "\n".join(relevant_knowledge)
        
        intermediate_responses = []
        
        # Process through expert chain
        for model_name, expert in self.experts.items():
            # Different process method for Gemini
            if isinstance(expert, GeminiExpert):
                response = expert.process(query, context)
            else:
                response = expert.process(query, context)
            
            response_entry = {
                "model": model_name,
                "role": self.default_roles.get(model_name, "Expert"),
                "response": response
            }
            
            intermediate_responses.append(response_entry)
            
            # Add to context manager
            self.context_manager.add_context(model_name, response_entry['role'], response)
        
        # Propagate knowledge across experts
        self.knowledge_transfer.propagate_knowledge(intermediate_responses)
        
        # Aggregate responses (similar to previous implementation)
        aggregator = Expert(
            self.available_models[aggregator_model],
            "Response Aggregator"
        )
        
        aggregator_prompt = PromptTemplate(
            input_variables=["responses", "query"],
            template="""Synthesize these expert perspectives into a comprehensive response:
            Original Query: {query}
            Expert Responses: {responses}
            
            Provide a nuanced, well-rounded answer that integrates insights from multiple experts."""
        )
        
        chain = LLMChain(llm=aggregator.llm, prompt=aggregator_prompt)
        final_response = chain.run(
            responses=json.dumps(intermediate_responses, indent=2),
            query=query
        )
        
        return final_response, intermediate_responses
        
    
    def auto_select_experts(self, query: str) -> Tuple[List[str], Dict[float, str]]:
        scores = self.gating.analyze_query(query)
        # Select experts with scores above threshold (0.3)
        selected = [model for model, score in scores.items() if score > 0.3]
        return selected, scores
    
def create_expert_radar_chart(scores: Dict[str, float]):
    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])  # Complete the circle
    categories.append(categories[0])  # Complete the circle
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Expert Relevance'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Expert Relevance Analysis"
    )
    return fig

def create_response_flow_diagram(intermediate_responses: List[Dict]):
    fig = make_subplots(rows=len(intermediate_responses), cols=1, 
                       subplot_titles=[f"{resp['model'].title()} ({resp['role']})" 
                                     for resp in intermediate_responses])
    
    for i, resp in enumerate(intermediate_responses, 1):
        response_len = len(resp['response'])
        fig.add_trace(
            go.Bar(
                x=[response_len],
                y=[resp['model']],
                orientation='h',
                name=resp['model'],
                text=[f"{response_len} chars"],
                textposition='auto',
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=100 * len(intermediate_responses),
        title_text="Response Contributions",
        showlegend=False
    )
    return fig

def main():
    st.title("CoRE: Collaboration of Role-based Experts")
    st.subheader("Configure the Experts in the sidebar")
    
    # Sidebar configuration
    st.sidebar.title("System Configuration")
    # Modify main function to include Gemini API key input
    st.sidebar.header("Gemini API Configuration")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    # Initialize MoRE system with optional Gemini API key
    more_system = MoRESystem(gemini_api_key=gemini_api_key)
    
    # Expert Selection Method
    selection_method = st.sidebar.radio(
        "Expert Selection Method",
        ["Automatic", "Manual"]
    )
    
    selected_models = []
    custom_roles = {}
    
    with st.sidebar.expander("Expert Selection", expanded=True):
        if selection_method == "Manual":
            selected_models = st.multiselect(
                "Select Expert Models",
                list(more_system.available_models.keys()),
                default=["gemma", "llama"]
            )
        
    with st.sidebar.expander("Role Assignment", expanded=True):
        use_custom_roles = st.checkbox("Customize Expert Roles")
        if use_custom_roles and selection_method == "Manual":
            for model in selected_models:
                # Use a more explicit input with placeholder from default role
                custom_roles[model] = st.text_input(
                    f"Role for {model}",
                    value=more_system.default_roles.get(model, f"{model.capitalize()} Expert"),
                    key=f"role_{model}"  # Add unique key to prevent Streamlit caching issues
                )
    
    # Main content area
    col2, col1 = st.columns([1, 2])
    

    with col1:

        st.subheader("Query Input")
        query = st.text_area("Enter your query", height=100)
        
        if selection_method == "Automatic" and query:
            selected_models, expert_scores = more_system.auto_select_experts(query)
            st.write("Selected Experts:", ", ".join(selected_models))
            
            # Display radar chart of expert scores
            radar_chart = create_expert_radar_chart(expert_scores)
            st.plotly_chart(radar_chart)
    
    with col2:
        st.subheader("Aggregator Selection")
        aggregator_model = st.selectbox(
            "Select Aggregator Model",
            list(more_system.available_models.keys())
        )
    
    if st.button("Process Query", type="primary") and query and selected_models:
        try:
            with st.spinner("Processing query through expert chain..."):
                # Initialize experts with selected models and roles
                more_system.initialize_experts(selected_models, custom_roles if use_custom_roles else None)
                
                # Process query
                final_response, intermediate_responses = more_system.process_query(query, aggregator_model)
                
                # Display results
                st.header("Results")
                
                # Show response flow visualization
                flow_chart = create_response_flow_diagram(intermediate_responses)
                st.plotly_chart(flow_chart)
                
                # Show intermediate responses in an expander
                with st.expander("Expert Responses", expanded=False):
                    for resp in intermediate_responses:
                        st.subheader(f"{resp['model'].title()} ({resp['role']})")
                        st.write(resp['response'])
                
                # Show final response
                st.header("Final Aggregated Response")
                st.write(final_response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()