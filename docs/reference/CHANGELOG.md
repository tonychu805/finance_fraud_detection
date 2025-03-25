# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### 2024-03-24
- Reorganized documentation structure for better maintainability
- Added daily changelog format for easier tracking
- Updated model card with performance history table

## [1.0.0] - 2024-03-24
### Added
- Initial release of fraud detection system
- LightGBM and Random Forest models with probability calibration
- Optimal threshold finding for both models
- Basic feature engineering pipeline
- API endpoints for single and batch predictions
- Webhook support for fraud alerts
- Model monitoring and logging
- Documentation structure

### Model Performance
- LightGBM:
  - F1 Score: 0.7292 (threshold: 0.2071)
  - ROC AUC: 0.9961
  - Average Precision: 0.7422

- Random Forest:
  - F1 Score: 0.7567 (threshold: 0.0994)
  - ROC AUC: 0.9971
  - Average Precision: 0.7684

## [0.1.0] - 2024-03-19
### Added
- Project initialization
- Basic project structure
- Development environment setup
- Initial documentation framework
- GitHub Actions workflow
- Docker configuration
- Test framework setup

### Changed
- None (initial release)

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None

---
## Daily Update Template
Copy and paste this template for daily updates:

```markdown
### YYYY-MM-DD
#### Changes
- [Category] Description of change
- [Model] Description of model update
- [Data] Description of data change

#### Performance
- Model: [model name]
- Metrics: [key metrics]
- Notes: [brief notes]
``` 