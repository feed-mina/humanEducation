import os
from dotenv import load_dotenv
from agents.graph import create_graph

# Load environment variables
load_dotenv()

def test_story_generation():
    print("Initializing graph...")
    graph = create_graph()
    
    user_input = "미래 도시를 배경으로 한 짧은 SF 소설을 써줘."
    print(f"User Input: {user_input}")
    
    initial_state = {
        "messages": [user_input],
        "max_revisions": 1, # Limit revisions for faster testing
        "revision_count": 0
    }
    
    print("Invoking graph...")
    try:
        result = graph.invoke(initial_state)
        
        print("\n--- Final Result ---")
        print(f"Plan:\n{result.get('plan')}")
        print(f"\nDraft:\n{result.get('draft')}")
        print(f"\nCritique:\n{result.get('critique')}")
        
        print("\nTest Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
    else:
        test_story_generation()
