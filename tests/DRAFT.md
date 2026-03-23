# tests/

Organized by submodule, mirroring the source structure.

## Planned structure

### tests/spec/
- Test tarball extraction safety (path traversal, symlinks, bombs, permissions)
- Test parser against example agent repos (valid and invalid)
- Test validator catches bad skill names, missing fields, duplicates
- Fixture: minimal valid agent repo directory

### tests/server/
- Test API endpoints (create, get, delete) via FastAPI test client
- Test service layer with real SQLite DB
- Test bundle storage (store, retrieve, delete)
- Test name uniqueness constraint (409 on duplicate)
- Test deletion cleans up files and DB
- Fixture: sample tarballs of agent repos

### tests/client/
- Test CLI commands against a mock/test server
- Test client library methods
- Test error handling (server down, 404, 409)
