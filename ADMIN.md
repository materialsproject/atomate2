# Releasing a new atomate2 version

Version releases on Pypi and GitHub are handled automatically through GitHub
actions. The steps to push a new release are:

1. Update `CHANGELOG.md` with a new version and release notes.
2. Create a tagged Git commit with the above changes: `git tag v0.0.1`
3. Push the commit and tags to GitHub using: `git push origin --tags`
