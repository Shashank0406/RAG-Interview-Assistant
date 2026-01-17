"""
Streamlit Web Interface for RAG System
A beautiful and interactive interface for the Gen AI Interview RAG chatbot
"""

import streamlit as st
import sys
from pathlib import Path
import os
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system import create_rag_system
from config.settings import RAG_CONFIG
from conversation_memory import create_memory_system
from model_comparison import create_model_comparator
from evaluation import evaluate_rag_system, RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="Gen AI Interview Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .response-box {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1e3c72;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #28a745;
    }
    .stats-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the RAG system with proper configuration"""
    if 'rag_system' not in st.session_state:
        try:
            # Get API keys from environment or user input
            openai_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key", "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_key", "")

            st.session_state.rag_system = create_rag_system(
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                persist_directory=RAG_CONFIG["persist_directory"]
            )

            # Ingest sample documents if not already done
            if st.session_state.rag_system.get_stats()["total_documents"] == 0:
                sample_docs_path = Path(__file__).parent.parent / "data" / "sample_docs"
                if sample_docs_path.exists():
                    with st.spinner("Ingesting knowledge base..."):
                        st.session_state.rag_system.ingest_documents([str(sample_docs_path)])
                        st.success("Knowledge base loaded successfully!")

        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            return False

    return True

def initialize_conversation():
    """Initialize or get conversation session"""
    if 'conversation_id' not in st.session_state:
        # Create a new conversation session
        user_id = st.session_state.get("user_id", "default_user")
        topic = "Gen AI Interview Preparation"
        st.session_state.conversation_id = st.session_state.rag_system.start_conversation(user_id, topic)

    return st.session_state.conversation_id

def display_stats():
    """Display system statistics"""
    if 'rag_system' in st.session_state:
        stats = st.session_state.rag_system.get_stats()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", stats.get("total_documents", 0))
        with col2:
            st.metric("Embedding Model", RAG_CONFIG["embedding_model"].split("/")[-1])
        with col3:
            providers = len(stats.get("available_providers", []))
            st.metric("LLM Providers", providers)

def display_response(response):
    """Display the RAG response in a formatted way"""
    st.markdown('<div class="response-box">', unsafe_allow_html=True)

    # Answer section
    st.markdown("### Answer")
    st.write(response.get("answer", "No answer available"))

    # Confidence score
    confidence = response.get("confidence", 0.0)
    confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
    st.markdown(f"**Confidence Score:** {confidence_color} {confidence:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Sources section
    if response.get("sources"):
        st.markdown("### Sources")
        for i, source in enumerate(response["sources"], 1):
            with st.expander(f"Source {i}: {source['source']}"):
                st.write(source["content"])

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🤖 Gen AI Interview Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Your AI-powered study companion for Generative AI interviews*")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # API Keys section
        st.subheader("API Keys")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.get("openai_key", ""),
            help="Enter your OpenAI API key for GPT models"
        )
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_key", ""),
            help="Enter your Anthropic API key for Claude models"
        )

        # Update session state
        st.session_state.openai_key = openai_key
        st.session_state.anthropic_key = anthropic_key

        # Model selection
        st.subheader("Model Selection")
        available_providers = []
        if openai_key:
            available_providers.extend(["openai"])
        if anthropic_key:
            available_providers.extend(["anthropic"])

        if available_providers:
            selected_provider = st.selectbox(
                "Choose LLM Provider",
                available_providers,
                help="Select which AI model to use for answering questions"
            )
            st.session_state.selected_provider = selected_provider
        else:
            st.warning("⚠️ Add API keys to enable LLM-powered responses")
            st.session_state.selected_provider = None

        # System stats
        st.subheader("System Stats")
        display_stats()

        # Conversation Management
        st.subheader("💬 Conversation")
        if st.button("🆕 New Conversation"):
            if 'conversation_id' in st.session_state:
                del st.session_state.conversation_id
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
            st.success("Started new conversation!")
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

        # Show conversation stats if available
        if 'conversation_id' in st.session_state and 'rag_system' in st.session_state:
            try:
                conv_stats = st.session_state.rag_system.get_conversation_stats(st.session_state.conversation_id)
                st.metric("Messages", conv_stats.get("message_count", 0))
            except:
                pass

        # Actions
        st.subheader("🔧 System Actions")
        if st.button("🔄 Reload Knowledge Base"):
            if 'rag_system' in st.session_state:
                with st.spinner("Reloading knowledge base..."):
                    st.session_state.rag_system.clear_knowledge_base()
                    sample_docs_path = Path(__file__).parent.parent / "data" / "sample_docs"
                    if sample_docs_path.exists():
                        st.session_state.rag_system.ingest_documents([str(sample_docs_path)])
                        st.success("Knowledge base reloaded!")
                        st.rerun()

    # Initialize RAG system
    if not initialize_rag_system():
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        return

    # Main content area
    st.header("💬 Ask Questions")

    # Query input
    with st.container():
        st.markdown('<div class="query-box">', unsafe_allow_html=True)

        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., What is RAG? Explain the difference between supervised and unsupervised learning...",
            height=100,
            key="query_input"
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            ask_button = st.button("🚀 Ask", use_container_width=True, type="primary")
        with col2:
            clear_button = st.button("🧹 Clear Chat", use_container_width=True)
        with col3:
            # Show current provider
            if st.session_state.get("selected_provider"):
                st.info(f"Using: {st.session_state.selected_provider.upper()}")
            else:
                st.warning("Retrieval-only mode (add API keys for AI responses)")

        st.markdown('</div>', unsafe_allow_html=True)

    # Handle clear chat
    if clear_button:
        if 'chat_history' in st.session_state:
            st.session_state.chat_history = []
        st.rerun()

    # Handle query
    if ask_button and query.strip():
        with st.spinner("Thinking..."):
            try:
                # Initialize conversation if needed
                conversation_id = initialize_conversation()

                # Get response from RAG system with conversation context
                provider = st.session_state.get("selected_provider", "openai")
                response = st.session_state.rag_system.query(
                    query,
                    provider=provider,
                    session_id=conversation_id
                )

                # Store last response for evaluation
                st.session_state.last_query = query
                st.session_state.last_response = response

                # Add to chat history
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []

                st.session_state.chat_history.append({
                    "timestamp": datetime.now(),
                    "query": query,
                    "response": response,
                    "conversation_id": conversation_id
                })

            except Exception as e:
                st.error(f"Error processing query: {e}")

    # Display chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("📝 Conversation History")

        for i, chat_entry in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown("---")

                # Query
                st.markdown(f"**You:** {chat_entry['query']}")

                # Response
                display_response(chat_entry['response'])

                # Timestamp
                st.caption(f"Asked at: {chat_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

    # Sample questions
    with st.expander("💡 Sample Questions to Try"):
        sample_questions = [
            "What is RAG and how does it work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the main components of a neural network?",
            "How do you evaluate generative AI models?",
            "What are some popular prompt engineering techniques?",
            "Explain the concept of embeddings in machine learning",
            "What are the ethical considerations in generative AI?",
            "How does fine-tuning work for large language models?",
        ]

        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}"):
                st.session_state.query_input = question
                st.rerun()

    # Model Comparison Section
    st.header("🔄 Model Comparison")
    st.markdown("Compare different AI models side by side on the same questions.")

    # Model comparison interface
    if st.session_state.get("selected_provider"):
        comparator = create_model_comparator(st.session_state.rag_system)

        # Get available models
        available_models = comparator.get_available_models()
        model_names = [m["name"] for m in available_models]

        if len(model_names) > 1:
            col1, col2 = st.columns([2, 1])

            with col1:
                comparison_query = st.text_area(
                    "Comparison Query:",
                    placeholder="Enter a question to compare models...",
                    key="comparison_query",
                    height=100
                )

            with col2:
                selected_models = st.multiselect(
                    "Select models to compare:",
                    model_names,
                    default=model_names[:2],  # Default to first 2 models
                    key="selected_models"
                )

                compare_button = st.button("⚖️ Compare Models", use_container_width=True, type="secondary")

            if compare_button and comparison_query.strip() and selected_models:
                with st.spinner("Comparing models..."):
                    try:
                        # Run comparison
                        result = comparator.compare_models(
                            comparison_query,
                            models_to_compare=selected_models
                        )

                        # Display results
                        st.subheader("📊 Comparison Results")

                        # Winner announcement
                        if result.winner:
                            st.success(f"🏆 **Winner: {result.winner}**")

                        # Model results in columns
                        cols = st.columns(min(len(result.model_results), 3))

                        for i, model_result in enumerate(result.model_results):
                            with cols[i % len(cols)]:
                                with st.container():
                                    st.markdown(f"### {model_result.model_name}")

                                    if model_result.error:
                                        st.error(f"❌ Error: {model_result.error}")
                                    else:
                                        # Metrics
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.metric("Response Time", f"{model_result.response_time}s")
                                        with col_b:
                                            st.metric("Confidence", f"{model_result.confidence_score:.2f}")

                                        st.metric("Sources Used", model_result.sources_used)

                                        # Response
                                        with st.expander("Response", expanded=True):
                                            st.write(model_result.response)

                        # Overall metrics
                        st.subheader("📈 Overall Metrics")
                        metrics = result.comparison_metrics

                        if metrics:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Response Time", f"{metrics.get('average_response_time', 0):.2f}s")
                            with col2:
                                st.metric("Avg Confidence", f"{metrics.get('average_confidence', 0):.2f}")
                            with col3:
                                st.metric("Successful Queries", f"{metrics.get('successful_queries', 0)}/{metrics.get('total_models', 0)}")

                    except Exception as e:
                        st.error(f"Error during model comparison: {e}")
        else:
            st.info("🤖 Need at least 2 models to compare. Add more API keys!")
    else:
        st.info("🔑 Add API keys above to enable model comparison.")

    # Evaluation Section
    st.header("📊 System Evaluation")
    st.markdown("Evaluate your RAG system's performance with comprehensive metrics.")

    if st.button("🔬 Run Full Evaluation", use_container_width=True, type="primary"):
        with st.spinner("Evaluating RAG system performance..."):
            try:
                # Run evaluation
                evaluation_report = evaluate_rag_system(st.session_state.rag_system)

                # Display results
                st.success("Evaluation completed!")

                # Summary metrics
                summary = evaluation_report["summary"]
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Average Score", f"{summary['average_score']:.2f}")
                with col2:
                    st.metric("Median Score", f"{summary['median_score']:.2f}")
                with col3:
                    st.metric("Test Cases", summary['total_test_cases'])
                with col4:
                    st.metric("Grade Distribution", f"A: {summary['grade_distribution'].get('A', 0)}")

                # Detailed metrics
                st.subheader("📈 Detailed Metrics")
                metrics = evaluation_report["metric_averages"]

                if metrics:
                    cols = st.columns(3)
                    for i, (metric_name, value) in enumerate(metrics.items()):
                        with cols[i % 3]:
                            st.metric(
                                metric_name.replace('_', ' ').title(),
                                f"{value:.2f}"
                            )

                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = evaluation_report["recommendations"]

                if recommendations:
                    for rec in recommendations:
                        st.info(f"• {rec}")
                else:
                    st.success("Your system is performing well! 🎉")

                # Detailed results
                with st.expander("🔍 Detailed Results", expanded=False):
                    for i, result in enumerate(evaluation_report["detailed_results"]):
                        st.markdown(f"**Query {i+1}:** {result['query']}")
                        st.markdown(f"**Grade:** {result['grade']} (Score: {result['overall_score']:.2f})")

                        if result['feedback']:
                            st.markdown("**Feedback:**")
                            for feedback in result['feedback']:
                                st.markdown(f"- {feedback}")

                        st.markdown("---")

            except Exception as e:
                st.error(f"Error during evaluation: {e}")

    # Quick evaluation for current response
    if 'last_response' in st.session_state:
        st.subheader("🎯 Evaluate Last Response")

        if st.button("📝 Evaluate Current Answer", use_container_width=True):
            try:
                evaluator = RAGEvaluator()
                last_query = st.session_state.get('last_query', '')
                last_response = st.session_state.last_response

                # Extract sources from last response
                sources = last_response.get('sources', [])

                # Evaluate
                result = evaluator.evaluate_response(last_query, last_response.get('answer', ''), sources)

                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Score", f"{result.overall_score:.2f}")
                with col2:
                    st.metric("Grade", result.grade)
                with col3:
                    st.metric("Confidence", f"{last_response.get('confidence', 0):.2f}")

                # Feedback
                if result.feedback:
                    st.subheader("💬 Feedback")
                    for feedback in result.feedback:
                        st.write(f"• {feedback}")

                # Detailed metrics
                with st.expander("📊 Detailed Metrics"):
                    for metric_name, value in result.metrics.items():
                        st.write(f"**{metric_name.replace('_', ' ').title()}:** {value:.2f}")

            except Exception as e:
                st.error(f"Error evaluating response: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Built with Streamlit, LangChain, and ChromaDB • Ready for Gen AI Interviews!*")

# Health check endpoint for Docker
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check if RAG system is initialized
        rag_available = 'rag_system' in st.session_state

        # Get basic system stats
        stats = {}
        if rag_available:
            try:
                stats = st.session_state.rag_system.get_stats()
            except:
                stats = {"status": "rag_system_error"}

        health_data = {
            "status": "healthy" if rag_available else "initializing",
            "timestamp": datetime.now().isoformat(),
            "rag_system": rag_available,
            "stats": stats
        }

        return json.dumps(health_data)

    except Exception as e:
        error_data = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_data)

# Handle health check requests
if __name__ == "__main__":
    # Check for health endpoint
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        print(health_check())
    else:
        main()