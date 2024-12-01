# Leetcode_practise
This is my learning note of Leetcode problems.

+ How to update the recode
  - create a repository
  - VS code terminal: git clone https://github.com/你的用户名/my-new-project.git
  - VS code terminal: cd my-new-project
  - VS code terminal: git config --global user.name "Your Name"
                      git config --global user.email "youremail@example.com"
  - vscode : edit your file and save
  - first push:
    git status
    git add .
    git commit -m "Initial commit"
    git push origin main
  - later push:
    git add .
    git commit -m "updata"
    git push
  - To create a new branch:
    git checkout -b new-feature
    git push origin new-feature
  -vs code has a git panel to see the history you have done


    


  # how to cooperate with team
  + The leader:
    - One person creates a repository on GitHub
    - Add team members as collaborators by their GitHub usernames or email addresses.
    - check the main branch
  + The members
  - Clone the Repository : git clone <repository-url>
  - Each team member creates their own branch for new features or fixes: git checkout -b feature-branch-name
  - Team members make changes locally, test the code, and commit updates with clear commit messages: 
      git add .
      git commit -m "Describe the changes made"
  -Push changes to the respective branch on GitHub:
  git push origin feature-branch-name
+ update to main branch:
    When a feature or fix is ready, open a pull request to merge it into the main branch.
    - Go to the repository on GitHub.
    - Click Pull Requests > New Pull Request.
    - Compare your branch with the main branch and create the pull request.
    - Team members review the pull request, suggest changes, or approve it
    - After approval, merge the pull request into the main branch.
    - keep your local repository up to date with the main branch: 
       git pull origin main

# using your ternimal to create a branch and code:
- git status
git checkout -B chad/hello-world: it will switched to and reset branche"chad/hello-world"
touch HelloWorld.py
code HelloWorld.py
git status


