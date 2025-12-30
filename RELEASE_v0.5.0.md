# Release v0.5.0 - Canonical Volume Persistence & 3D Processing

## Major Features

### Canonical Volume Persistence
- **NRRD Format Export**: Processed volumes are now exported as standardized NRRD files with full metadata preservation
- **JSON Provenance Sidecar**: Complete provenance tracking including intensity bins, spacing, volume shape, and processing parameters
- **Reproducible Workflows**: All processing steps and configurations are saved for full reproducibility

### New Tab 3: 3D Processing
- **Interactive 3D Viewer**: Integrated PyVistaQt-based 3D visualization with orbit/pan/zoom controls
- **Threshold-Based Mesh Generation**: Generate 3D meshes from canonical volumes using intensity thresholds
- **Automatic Mesh Generation**: Meshes are automatically generated when loading canonical volumes
- **Vertex Coloring**: Meshes display with vertex colors mapped from intensity bins, enabling multi-colored visualization of different tissue types in a single mesh
- **Mesh Controls**: Opacity slider with manual input, smooth shading, edge display, and camera controls

## Improvements

- **Unified Loading Workflow**: Single progress dialog for combined volume loading and mesh generation
- **Enhanced Progress Tracking**: Phase-aware progress dialogs for NRRD export and mesh generation operations
- **Improved UI Responsiveness**: Background thread processing prevents UI freezing during volume operations
- **Bin Color Export**: Intensity bin colors are now properly exported to provenance metadata

## Technical Details

- Added `pyvista` and `pyvistaqt` dependencies for 3D visualization
- Canonical volumes stored with `int16` data type by default (with `float32` fallback)
- Spacing, origin, and direction metadata preserved in NRRD format
- Vertex color sampling from volume intensities at mesh vertices
- Worker lifecycle management to prevent thread destruction errors

## Files Changed

- New: `src/ct23d/gui/processing3d/` - 3D Processing tab implementation
- New: `scripts/view_nrrd.py` - Standalone NRRD file viewer utility
- Updated: `src/ct23d/core/volume.py` - Canonical volume persistence (save/load NRRD)
- Updated: `src/ct23d/gui/mesher/wizard.py` - Export workflow refactored for NRRD-only export
- Updated: `src/ct23d/gui/status.py` - Enhanced progress dialog phase tracking

## Upgrade Notes

This release introduces the canonical volume format as the primary persistence mechanism. Processed volumes should now be exported as NRRD files for use in the 3D Processing tab.

---

**Full Changelog**: See commit history for detailed changes

