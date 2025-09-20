import re

class MultiStepReasoner:
    """
    Break complex queries into subtasks, support follow-up queries, and explain reasoning.
    """
    def break_down_query(self, query: str):
        # very simple: split by 'and', ',', ';'
        delimiters = [",", ";", " and "]
        for d in delimiters:
            if d in query.lower():
                return [q.strip() for q in query.split(d)]
        return [query]

    def explain_reasoning(self, task_list):
        explanation = "Query broken down into the following subtasks:\n"
        for i, t in enumerate(task_list):
            explanation += f"{i+1}. {t}\n"
        return explanation
