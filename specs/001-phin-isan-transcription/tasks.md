---

description: "Task list for Thai Isan Lute (Phin) Music Transcription implementation"
---

# Tasks: Thai Isan Lute (Phin) Music Transcription

**Input**: Design documents from `/specs/001-phin-isan-transcription/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included as explicitly requested in the feature specification and required for research reproducibility.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- **Web app**: `backend/src/`, `frontend/src/`
- **Mobile**: `api/src/`, `ios/src/` or `android/src/`
- Paths shown below assume single project - adjust based on plan.md structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan in src/
- [ ] T002 [P] Initialize Python project with pyproject.toml and poetry.lock
- [ ] T003 [P] Configure directory structure: audio_sources/, sheet_music/, output/, references/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 [P] Install all required Python dependencies (librosa, pretty_midi, yt-dlp, torch, mir_eval, numpy, scipy) as specified in plan.md
- [ ] T005 [P] Create base configuration and constants module in src/utils/constants.py for Thai 7-tone scale system
- [ ] T006 [P] Create base utility functions in src/utils/__init__.py
- [ ] T007 Create base model classes in src/models/__init__.py
- [ ] T008 Create base data pipeline classes in src/data_pipeline/__init__.py
- [ ] T009 Create base evaluation classes in src/evaluation/__init__.py
- [ ] T010 Create base setup classes in src/setup/__init__.py
- [ ] T011 Setup pytest configuration in pyproject.toml with audio validation tests
- [ ] T012 [P] Create GPU detection and verification module in src/setup/environment.py
- [ ] T013 [P] Implement reproducibility framework with fixed random seeds configuration in src/utils/reproducibility.py
- [ ] T014 [P] Create experiment logging and documentation framework in src/utils/experiment_logger.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Environment Setup and Dependencies Installation (Priority: P1) 🎯 MVP

**Goal**: Create a complete Python environment with all necessary dependencies for Thai Isan lute music transcription, including audio processing libraries, machine learning frameworks, and YouTube download tools.

**Independent Test**: Can be fully tested by running the setup script and verifying that all required libraries (librosa, pretty_midi, torch/tensorflow, mir_eval, yt-dlp) are successfully installed and accessible.

### Tests for User Story 1 (OPTIONAL - only if tests requested) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T015 [P] [US1] Unit test for dependency installation verification in tests/unit/test_setup.py
- [ ] T016 [P] [US1] Integration test for environment validation in tests/integration/test_environment.py

### Implementation for User Story 1

- [ ] T017 [P] [US1] Create environment setup module in src/setup/environment.py with dependency installation and GPU detection
- [ ] T018 [US1] Create directory structure initialization in src/setup/environment.py (audio_sources/, sheet_music/, etc.)
- [ ] T019 [US1] Implement dependency verification functions to confirm all required libraries are accessible
- [ ] T020 [US1] Add logging for environment setup operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Thai Isan Music Data Collection and Preprocessing (Priority: P2)

**Goal**: Create functionality to automatically download authentic Thai Isan music videos from YouTube and convert them to audio files suitable for transcription model training, following the traditional 7-tone musical system.

**Independent Test**: Can be tested by running the batch download script with YouTube URLs and verifying that the audio files are downloaded, converted to .wav format, and stored in the correct directory structure.

### Tests for User Story 2 (OPTIONAL - only if tests requested) ⚠️

- [ ] T021 [P] [US2] Unit test for YouTube audio download functionality in tests/unit/test_download.py
- [ ] T022 [P] [US2] Unit test for CQT feature extraction in tests/unit/test_feature_extraction.py

### Implementation for User Story 2

- [ ] T023 [P] [US2] Create YouTube download module in src/data_pipeline/download.py for downloading Isan music videos
- [ ] T024 [P] [US2] Create CQT feature extraction module in src/data_pipeline/feature_extraction.py with extract_phin_features function
- [ ] T025 [US2] Implement audio conversion from YouTube to .wav format in src/data_pipeline/download.py
- [ ] T026 [US2] Implement proper audio normalization for Thai Isan music characteristics in src/data_pipeline/feature_extraction.py
- [ ] T027 [US2] Implement CQT spectrogram normalization to ensure compatibility with model input in src/data_pipeline/feature_extraction.py
- [ ] T028 [US2] Create AudioSource entity model in src/data_pipeline/models.py based on data model specification
- [ ] T029 [US2] Add file path management and organization for audio_sources/ directory

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Music Transcription Model Framework (Priority: P3)

**Goal**: Create a basic model architecture that can transcribe Thai Isan lute music using Constant-Q Transform features and appropriate evaluation metrics to measure transcription accuracy.

**Independent Test**: Can be tested by running the evaluation function on sample transcriptions and verifying that Onset F1 and Pitch F1 metrics are calculated correctly.

### Tests for User Story 3 (OPTIONAL - only if tests requested) ⚠️

- [ ] T031 [P] [US3] Unit test for PhinTranscriber model architecture in tests/unit/test_model_architecture.py
- [ ] T032 [P] [US3] Unit test for transcription evaluation functions in tests/unit/test_evaluation.py
- [ ] T033 [P] [US3] Integration test for complete transcription pipeline in tests/integration/test_transcription.py

### Implementation for User Story 3

- [ ] T034 [P] [US3] Create PhinTranscriber model class in src/models/phin_transcriber.py with CNN-RNN-Attention architecture
- [ ] T035 [P] [US3] Create transcription evaluation module in src/evaluation/transcription_eval.py with evaluate_transcription function
- [ ] T036 [US3] Implement CNN layers in PhinTranscriber model using PyTorch nn.Conv2d
- [ ] T037 [US3] Implement RNN layers in PhinTranscriber model using PyTorch nn.LSTM/nn.GRU
- [ ] T038 [US3] Implement attention mechanism in PhinTranscriber model
- [ ] T039 [US3] Create TranscriptionOutput entity model in src/models/transcription_output.py
- [ ] T040 [US3] Create CQTFeature entity model in src/data_pipeline/models.py
- [ ] T041 [US3] Create EvaluationResult entity model in src/evaluation/models.py
- [ ] T042 [US3] Implement evaluation metrics (Onset F1 and Pitch F1) using mir_eval library
- [ ] T043 [US3] Add model input validation to ensure CQT spectrograms are compatible

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: API Endpoints Implementation

**Goal**: Implement API endpoints for the transcription service

- [ ] T046 [P] Create API router structure in src/api/main.py
- [ ] T047 [P] Implement /transcribe endpoint in src/api/endpoints/transcription.py
- [ ] T048 [P] Implement /evaluate endpoint in src/api/endpoints/evaluation.py
- [ ] T049 [P] Implement /extract_features endpoint in src/api/endpoints/feature_extraction.py
- [ ] T050 Create API request/response models in src/api/models.py based on contract specification
- [ ] T051 Add API error handling and validation

**Checkpoint**: API endpoints are functional and integrate with the core components

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T052 [P] Documentation updates in docs/ and README.md
- [ ] T053 Code cleanup and refactoring
- [ ] T054 Performance optimization for CQT feature extraction
- [ ] T055 [P] Additional unit tests (if requested) in tests/unit/
- [ ] T056 Run quickstart.md validation to ensure all examples work correctly
- [ ] T057 Create main entry point for end-to-end pipeline in src/main.py
- [ ] T058 Add comprehensive logging throughout the application
- [ ] T059 Add configuration management for different environments
- [ ] T060 Create model card documentation for PhinTranscriber model following constitution requirements
- [ ] T061 Implement reproducibility checklist validation before merging

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **API Implementation (Phase 6)**: Depends on User Stories 1-3 completion
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on User Story 2 for CQT features

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Edge Case Handling

- [ ] T062 Create error handling for corrupted audio files in src/data_pipeline/error_handler.py
- [ ] T063 Implement recording environment detection and normalization in src/data_pipeline/preprocessing.py
- [ ] T064 Add fallback mechanism for complex polyphonic audio in src/models/phin_transcriber.py

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
T015 [P] [US1] Unit test for dependency installation verification in tests/unit/test_setup.py
T016 [P] [US1] Integration test for environment validation in tests/integration/test_environment.py

# Launch all implementation tasks for User Story 1 together:
T017 [P] [US1] Create environment setup module in src/setup/environment.py with dependency installation and GPU detection
T018 [US1] Create directory structure initialization in src/setup/environment.py (audio_sources/, sheet_music/, etc.)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add API Endpoints → Test integration → Deploy
6. Add Polish → Final validation → Production ready
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: API Implementation
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Total tasks: 64