"""Stage 2 on LangChain's prebuilt agent (create_agent) + Gemini. Same tools, calculator, gates and grading as the
LangGraph version (stage2_langraph_agent.py).

Run:  python -m ssa.stage2_prebuilt_agent                     # all 7 scenarios
      python -m ssa.stage2_prebuilt_agent gold-late-delivery  # just one
"""

import sys

from ssa.agent_prebuilt import run_prebuilt
from ssa.stage2_langraph_agent import main

if __name__ == "__main__":
    main(sys.argv[1:], runner=run_prebuilt, tag="stage2-prebuilt")
