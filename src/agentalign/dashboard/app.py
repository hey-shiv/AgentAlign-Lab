"""Dashboard application.

Main Streamlit app for visualizing results.
"""

import streamlit as st


def main():
    """Run dashboard app."""
    st.set_page_config(page_title="AgentAlign Dashboard", layout="wide")

    st.title("AgentAlign Lab Dashboard")
    st.write("Visualization of agent alignment experiments.")

    # Sidebar
    st.sidebar.header("Controls")
    page = st.sidebar.radio("Select Page", ["Overview", "Results", "Analysis"])

    if page == "Overview":
        st.header("Overview")
        st.write("Project overview and statistics.")

    elif page == "Results":
        st.header("Results")
        st.write("Experiment results and metrics.")

    elif page == "Analysis":
        st.header("Analysis")
        st.write("Detailed analysis and insights.")


if __name__ == "__main__":
    main()
