# research_agent/reasoning.py
class MultiStepReasoner:
    def break_down_query(self, query):
        # Simple split by sentences or semicolons as subtasks
        import re
        tasks = re.split(r'[.;]', query)
        tasks = [t.strip() for t in tasks if t.strip()]
        return tasks

    def explain_reasoning(self, tasks):
        explanation = "The query is broken down into subtasks to process each aspect separately:\n"
        for i, t in enumerate(tasks):
            explanation += f"{i+1}. {t}\n"
        return explanation
