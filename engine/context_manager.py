import os

class WorkspaceContextManager:
    """
    Scans and formats local workspace file context within budget constraints.
    """
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path

    def build_workspace_prompt(self, user_query: str) -> str:
        if not os.path.exists(self.workspace_path):
            return user_query

        file_tree = []
        file_contents = []
        
        readable_exts = {'.py', '.md', '.sql', '.json', '.txt'}
        
        for root, _, files in os.walk(self.workspace_path):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), self.workspace_path)
                file_tree.append(rel_path)
                
                ext = os.path.splitext(f)[1].lower()
                if ext in readable_exts and len(file_contents) < 4:
                    full_path = os.path.join(root, f)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as fp:
                            content = fp.read()[:800] # Trim per file to protect memory budget
                            file_contents.append(f"--- FILE: {rel_path} ---\n{content}")
                    except Exception:
                        pass

        context_str = f"Workspace Directory: '{self.workspace_path}'\n"
        context_str += "Files: " + ", ".join(file_tree[:15]) + "\n\n"
        context_str += "\n\n".join(file_contents)
        
        full_prompt = (
            f"<|im_start|>system\nYou are an AI code assistant analyzing a local workspace.\n"
            f"Codebase Context:\n{context_str}<|im_end|>\n"
            f"<|im_start|>user\n{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return full_prompt
