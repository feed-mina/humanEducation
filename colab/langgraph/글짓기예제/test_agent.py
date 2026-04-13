import os
from dotenv import load_dotenv
from agents.graph import create_graph

# Load environment variables
load_dotenv()

def test_agent():
    print("Initializing graph...")
    graph = create_graph()
    
    user_input = "Write a short poem about a coding AI agent."
    print(f"User Input: {user_input}")
    
    initial_state = {
        "messages": [user_input],
        "max_revisions": 2,
        "revision_count": 0
    }
    
    print("Invoking graph...")
    try:
        result = graph.invoke(initial_state)
        
        print("\n--- Final Result ---")
        print(f"Plan:\n{result.get('plan')}")
        print(f"\nDraft:\n{result.get('draft')}")
        print(f"\nRevision Count: {result.get('revision_count')}")
        
        print("\n--- Revisions History ---")
        for rev in result.get('revisions', []):
            print(f"[{rev.get('step')}] {rev.get('content')[:50]}...")
            
        print("\nTest Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set it in .env file.")
    else:
        test_agent()
