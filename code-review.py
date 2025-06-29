import re
import requests
from typing import Dict, List, Optional, TypedDict, Annotated
from urllib.parse import urlparse
import os
import json
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# State definition for the workflow
class PRReviewState(TypedDict):
    pr_url: str
    pr_number: int
    repo_owner: str
    repo_name: str
    files_data: List[Dict]
    review_instructions: Dict[str, str]
    review_comments: List[Dict]
    github_token: str
    messages: Annotated[list, add_messages]

@dataclass
class ReviewRule:
    pattern: str
    instructions: str
    file_types: List[str]

class GitHubPRReviewer:
    def __init__(self, github_token: str, openai_api_key: str):
        self.github_token = github_token
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
        
        # Define review rules with regex patterns and instructions
        self.review_rules = [
            ReviewRule(
                pattern=r"@Controller|@RestController",
                instructions="""
                For Spring Boot Controller classes, review for:
                1. Proper HTTP method annotations (@GetMapping, @PostMapping, etc.)
                2. Input validation using @Valid or @Validated
                3. Consistent error handling and response formats
                4. Proper use of ResponseEntity for status codes
                5. Security considerations for endpoints
                6. Request/Response body documentation
                7. Path variable and request parameter validation
                """,
                file_types=[".java"]
            ),
            ReviewRule(
                pattern=r"@Service|@Component",
                instructions="""
                For Spring Boot Service classes, review for:
                1. Proper transaction management (@Transactional)
                2. Exception handling and error propagation
                3. Business logic separation from controllers
                4. Dependency injection best practices
                5. Method naming conventions
                6. Null safety and validation
                """,
                file_types=[".java"]
            ),
            ReviewRule(
                pattern=r"@Repository|@Entity",
                instructions="""
                For Spring Boot Repository/Entity classes, review for:
                1. Proper JPA annotations and relationships
                2. Query optimization and N+1 problems
                3. Database indexing considerations
                4. Entity lifecycle callbacks
                5. Proper use of cascading operations
                6. Data validation constraints
                """,
                file_types=[".java"]
            ),
            ReviewRule(
                pattern=r"class.*React\.Component|function.*\{|const.*=.*\(",
                instructions="""
                For React components, review for:
                1. Proper use of hooks (useState, useEffect, etc.)
                2. Component composition and reusability
                3. Props validation with PropTypes or TypeScript
                4. Performance optimizations (useMemo, useCallback)
                5. Accessibility (a11y) considerations
                6. Error boundaries and error handling
                7. State management patterns
                """,
                file_types=[".js", ".jsx", ".ts", ".tsx"]
            ),
            ReviewRule(
                pattern=r"def.*\(|class.*:|async def",
                instructions="""
                For Python functions/classes, review for:
                1. Type hints and documentation
                2. Error handling with proper exceptions
                3. Code organization and single responsibility
                4. Performance considerations
                5. Security vulnerabilities
                6. Proper use of async/await patterns
                7. Memory management and resource cleanup
                """,
                file_types=[".py"]
            ),
            ReviewRule(
                pattern=r"SELECT|INSERT|UPDATE|DELETE",
                instructions="""
                For SQL queries, review for:
                1. SQL injection prevention
                2. Query performance and indexing
                3. Proper joins and relationships
                4. Data validation and constraints
                5. Transaction handling
                6. Error handling for database operations
                """,
                file_types=[".sql", ".java", ".py", ".js"]
            )
        ]

    def parse_pr_url(self, state: PRReviewState) -> PRReviewState:
        """Parse GitHub PR URL to extract owner, repo, and PR number"""
        try:
            # Example URL: https://github.com/owner/repo/pull/123
            url = state["pr_url"].strip('/')
            
            if 'github.com' not in url:
                raise ValueError("Invalid GitHub URL")
            
            # Handle both HTTP and HTTPS URLs
            if url.startswith('http'):
                parsed_url = urlparse(url)
                path_parts = parsed_url.path.strip('/').split('/')
            else:
                path_parts = url.split('/')
                
            # Validate URL structure
            if len(path_parts) < 4 or 'pull' not in path_parts:
                raise ValueError("URL must be in format: github.com/owner/repo/pull/number")
            
            # Find the indices for owner, repo, and PR number
            pull_index = path_parts.index('pull')
            if pull_index < 2:
                raise ValueError("Invalid GitHub PR URL structure")
                
            repo_owner = path_parts[pull_index - 2]
            repo_name = path_parts[pull_index - 1]
            pr_number = int(path_parts[pull_index + 1])
            
            state["repo_owner"] = repo_owner
            state["repo_name"] = repo_name
            state["pr_number"] = pr_number
            
            print(f"Parsed PR: {repo_owner}/{repo_name}#{pr_number}")
            return state
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse PR URL '{state['pr_url']}': {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing PR URL: {e}")

    def fetch_pr_files(self, state: PRReviewState) -> PRReviewState:
        """Fetch files changed in the PR"""
        url = f"https://api.github.com/repos/{state['repo_owner']}/{state['repo_name']}/pulls/{state['pr_number']}/files"
        headers = {
            "Authorization": f"token {state['github_token']}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Code-Reviewer/1.0"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            files_data = response.json()
            
            # Filter for relevant files and add patch info
            processed_files = []
            for file_data in files_data:
                # Skip binary files, deleted files, and files without patches
                if (file_data.get('status') in ['added', 'modified'] and 
                    file_data.get('patch') and 
                    not file_data.get('binary', False)):
                    
                    # Skip large files (GitHub API limit)
                    if file_data.get('changes', 0) > 1000:
                        print(f"Skipping large file: {file_data['filename']} ({file_data.get('changes')} changes)")
                        continue
                    
                    processed_files.append({
                        'filename': file_data['filename'],
                        'patch': file_data.get('patch', ''),
                        'additions': file_data.get('additions', 0),
                        'deletions': file_data.get('deletions', 0),
                        'changes': file_data.get('changes', 0),
                        'status': file_data['status']
                    })
            
            state["files_data"] = processed_files
            print(f"✅ Fetched {len(processed_files)} files for review")
            
            if not processed_files:
                print("⚠️ No files found to review (all files may be binary, deleted, or too large)")
            
            return state
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 404:
                    raise Exception(f"PR not found: {state['repo_owner']}/{state['repo_name']}#{state['pr_number']}")
                elif e.response.status_code == 401:
                    raise Exception("GitHub authentication failed - check your token")
                elif e.response.status_code == 403:
                    raise Exception("GitHub API rate limit exceeded or insufficient permissions")
                else:
                    raise Exception(f"GitHub API error {e.response.status_code}: {e.response.text}")
            else:
                raise Exception(f"Network error fetching PR files: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from GitHub API")
        except Exception as e:
            raise Exception(f"Unexpected error fetching PR files: {e}")

    def determine_review_instructions(self, state: PRReviewState) -> PRReviewState:
        """Determine which review instructions apply to each file"""
        file_instructions = {}
        
        for file_data in state["files_data"]:
            filename = file_data['filename']
            file_ext = '.' + filename.split('.')[-1] if '.' in filename else ''
            patch_content = file_data.get('patch', '')
            
            applicable_rules = []
            
            for rule in self.review_rules:
                # Check if file extension matches
                if file_ext in rule.file_types or any(ft in filename for ft in rule.file_types):
                    # Check if pattern matches in the patch content
                    if re.search(rule.pattern, patch_content, re.IGNORECASE):
                        applicable_rules.append(rule.instructions)
            
            if applicable_rules:
                file_instructions[filename] = "\n\n".join(applicable_rules)
            else:
                # Default general review instructions
                file_instructions[filename] = """
                General code review guidelines:
                1. Code readability and maintainability
                2. Error handling and edge cases
                3. Performance considerations
                4. Security vulnerabilities
                5. Code duplication and reusability
                6. Naming conventions and documentation
                """
        
        state["review_instructions"] = file_instructions
        print(f"Determined review instructions for {len(file_instructions)} files")
        return state

    def analyze_code_with_ai(self, state: PRReviewState) -> PRReviewState:
        """Use AI to analyze code and generate review comments"""
        review_comments = []
        
        for file_data in state["files_data"]:
            filename = file_data['filename']
            patch = file_data.get('patch', '')
            instructions = state["review_instructions"].get(filename, '')
            
            if not patch or not instructions:
                continue
            
            # Extract line numbers from patch for better accuracy
            patch_lines = self._extract_patch_lines(patch)
            
            # Create prompt for AI analysis
            prompt = f"""
            You are an expert code reviewer. Analyze the following code changes and provide specific, actionable feedback.
            
            File: {filename}
            
            Review Instructions:
            {instructions}
            
            Code Changes (Git Patch):
            {patch}
            
            Available line numbers in this patch: {patch_lines}
            
            IMPORTANT: Return ONLY a valid JSON array. No additional text or markdown formatting.
            
            Format:
            [
                {{
                    "line": <line_number_from_patch>,
                    "comment": "<specific review comment>",
                    "severity": "high|medium|low"
                }}
            ]
            
            Rules:
            - Only comment on lines that have actual issues or improvements
            - Use line numbers that exist in the patch: {patch_lines}
            - Provide constructive, specific feedback
            - Be concise but helpful
            - Return empty array [] if no issues found
            """
            
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                ai_response = response.content.strip()
                
                # Clean response - remove markdown formatting if present
                if ai_response.startswith('```'):
                    lines = ai_response.split('\n')
                    ai_response = '\n'.join(lines[1:-1])
                
                # Parse JSON response
                try:
                    file_comments = json.loads(ai_response)
                    
                    if isinstance(file_comments, list):
                        for comment in file_comments:
                            if isinstance(comment, dict) and all(k in comment for k in ['line', 'comment']):
                                line_num = comment.get('line')
                                # Validate line number exists in patch
                                if line_num in patch_lines:
                                    review_comments.append({
                                        'filename': filename,
                                        'line': line_num,
                                        'comment': comment.get('comment', '').strip(),
                                        'severity': comment.get('severity', 'medium')
                                    })
                                else:
                                    print(f"Warning: Line {line_num} not found in patch for {filename}")
                    else:
                        print(f"AI response is not a list for {filename}")
                        
                except json.JSONDecodeError as je:
                    print(f"JSON parse error for {filename}: {je}")
                    print(f"AI Response: {ai_response[:200]}...")
                    continue
                    
            except Exception as e:
                print(f"Failed to analyze {filename}: {e}")
                continue
        
        state["review_comments"] = review_comments
        print(f"Generated {len(review_comments)} review comments")
        return state
    
    def _extract_patch_lines(self, patch: str) -> List[int]:
        """Extract line numbers from git patch format"""
        lines = []
        current_line = 0
        
        for line in patch.split('\n'):
            if line.startswith('@@'):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                try:
                    parts = line.split(' ')
                    new_info = parts[2]  # +new_start,new_count
                    new_start = int(new_info.split(',')[0][1:])  # Remove '+' and get start
                    current_line = new_start
                except (IndexError, ValueError):
                    continue
            elif line.startswith('+') and not line.startswith('+++'):
                # This is a new line
                lines.append(current_line)
                current_line += 1
            elif not line.startswith('-'):
                # Context line or unchanged line
                current_line += 1
                
        return lines

    def post_review_comments(self, state: PRReviewState) -> PRReviewState:
        """Post review comments to GitHub PR"""
        headers = {
            "Authorization": f"token {state['github_token']}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Code-Reviewer/1.0"
        }
        
        successful_comments = 0
        failed_comments = []
        
        for comment_data in state["review_comments"]:
            if not all(k in comment_data for k in ['filename', 'line', 'comment']):
                print(f"Skipping invalid comment data: {comment_data}")
                continue
                
            # Post individual line comment
            comment_url = f"https://api.github.com/repos/{state['repo_owner']}/{state['repo_name']}/pulls/{state['pr_number']}/comments"
            
            # Create severity emoji
            severity_emoji = {
                'high': '🚨',
                'medium': '⚠️',
                'low': '💡'
            }.get(comment_data['severity'], '🤖')
            
            comment_payload = {
                "body": f"{severity_emoji} **AI Review ({comment_data['severity'].upper()})**: {comment_data['comment']}",
                "path": comment_data['filename'],
                "line": comment_data['line'],
                "side": "RIGHT"  # Comment on the new version
            }
            
            try:
                response = requests.post(comment_url, headers=headers, json=comment_payload, timeout=30)
                
                if response.status_code == 201:
                    successful_comments += 1
                    print(f"✅ Posted comment on {comment_data['filename']}:{comment_data['line']}")
                else:
                    error_msg = f"Failed to post comment on {comment_data['filename']}:{comment_data['line']} - {response.status_code}"
                    if response.status_code == 422:
                        # Likely a line number issue
                        error_msg += " (Line number may not exist in diff)"
                    elif response.status_code == 401:
                        error_msg += " (Authentication failed - check GitHub token)"
                    elif response.status_code == 403:
                        error_msg += " (Permission denied - token needs repo access)"
                    
                    print(f"❌ {error_msg}")
                    failed_comments.append({
                        'comment': comment_data,
                        'error': error_msg,
                        'status_code': response.status_code
                    })
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error posting comment on {comment_data['filename']}:{comment_data['line']}: {e}"
                print(f"❌ {error_msg}")
                failed_comments.append({
                    'comment': comment_data,
                    'error': error_msg,
                    'status_code': None
                })
        
        # Post overall review summary only if we have comments
        if state["review_comments"]:
            self._post_review_summary(state, headers, successful_comments, failed_comments)
        
        print(f"✅ Successfully posted {successful_comments} out of {len(state['review_comments'])} comments")
        if failed_comments:
            print(f"❌ Failed to post {len(failed_comments)} comments")
            
        return state
    
    def _post_review_summary(self, state: PRReviewState, headers: dict, successful_comments: int, failed_comments: list):
        """Post overall review summary"""
        review_url = f"https://api.github.com/repos/{state['repo_owner']}/{state['repo_name']}/pulls/{state['pr_number']}/reviews"
        
        total_comments = len(state["review_comments"])
        high_severity = len([c for c in state["review_comments"] if c.get('severity') == 'high'])
        medium_severity = len([c for c in state["review_comments"] if c.get('severity') == 'medium'])
        low_severity = len([c for c in state["review_comments"] if c.get('severity') == 'low'])
        
        # Create summary with more details
        summary_body = f"""🤖 **AI Code Review Summary**

📊 **Review Statistics:**
- Files reviewed: {len(state['files_data'])}
- Total issues found: {total_comments}
- Successfully posted: {successful_comments}
- Failed to post: {len(failed_comments)}

🎯 **Issue Breakdown:**
- 🚨 High priority: {high_severity}
- ⚠️ Medium priority: {medium_severity}
- 💡 Low priority: {low_severity}

Please review the individual line comments for detailed feedback."""
        
        # Add failed comments info if any
        if failed_comments:
            summary_body += f"\n\n⚠️ **Note:** {len(failed_comments)} comments could not be posted due to line number mismatches or API errors."
        
        # Determine review event based on severity
        if high_severity > 0:
            review_event = "REQUEST_CHANGES"
        elif medium_severity > 3:  # Too many medium issues
            review_event = "REQUEST_CHANGES"
        else:
            review_event = "COMMENT"
        
        review_payload = {
            "body": summary_body,
            "event": review_event
        }
        
        try:
            response = requests.post(review_url, headers=headers, json=review_payload, timeout=30)
            if response.status_code == 200:
                print("✅ Posted review summary successfully")
            else:
                print(f"❌ Failed to post review summary: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to post review summary due to network error: {e}")

    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(PRReviewState)
        
        # Add nodes
        workflow.add_node("parse_url", self.parse_pr_url)
        workflow.add_node("fetch_files", self.fetch_pr_files)
        workflow.add_node("determine_instructions", self.determine_review_instructions)
        workflow.add_node("analyze_code", self.analyze_code_with_ai)
        workflow.add_node("post_comments", self.post_review_comments)
        
        # Add edges
        workflow.set_entry_point("parse_url")
        workflow.add_edge("parse_url", "fetch_files")
        workflow.add_edge("fetch_files", "determine_instructions")
        workflow.add_edge("determine_instructions", "analyze_code")
        workflow.add_edge("analyze_code", "post_comments")
        workflow.add_edge("post_comments", END)
        
        return workflow.compile()

# Usage example
def run_pr_review(pr_url: str, github_token: str, openai_api_key: str):
    """Run the PR review workflow"""
    reviewer = GitHubPRReviewer(github_token, openai_api_key)
    workflow = reviewer.create_workflow()
    
    # Initial state
    initial_state = {
        "pr_url": pr_url,
        "github_token": github_token,
        "pr_number": 0,
        "repo_owner": "",
        "repo_name": "",
        "files_data": [],
        "review_instructions": {},
        "review_comments": [],
        "messages": []
    }
    
    try:
        # Run the workflow
        result = workflow.invoke(initial_state)
        print("PR review completed successfully!")
        return result
    except Exception as e:
        print(f"PR review failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Set your tokens
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Example PR URL
    PR_URL = "https://github.com/your-org/your-repo/pull/123"
    
    if GITHUB_TOKEN and OPENAI_API_KEY:
        run_pr_review(PR_URL, GITHUB_TOKEN, OPENAI_API_KEY)
    else:
        print("Please set GITHUB_TOKEN and OPENAI_API_KEY environment variables")
