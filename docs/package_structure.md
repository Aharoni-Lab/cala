# Filetree

```
streaming_service/  
   ├── core/  
   │   ├── base.py            # Main base class (high-level orchestration only)  
   │   ├── estimates.py       # Results/estimates data structure  
   │   └── parameters.py      # Configuration parameters  
   ├── components/  
   │   ├── component.py       # Base Component class  
   │   ├── spatial.py         # Spatial component management  
   │   ├── temporal.py        # Temporal component management  
   │   └── background.py      # Background component handling  
   ├── processing/  
   │   ├── motion.py          # Motion correction  
   │   ├── initialization.py  # Different initialization strategies  
   │   ├── update.py          # Component update algorithms  
   │   └── deconvolution.py   # Spike deconvolution  
   ├── detection/  
   │   ├── candidate.py       # New component detection  
   │   ├── quality.py         # Component quality assessment  
   │   └── merger.py          # Component merging  
   └── utils/  
       ├── visualization.py   # Visualization tools  
       ├── buffer.py          # RingBuffer implementation  
       └── matrix_ops.py      # Matrix operations  
```

# Descriptions

1. Core Classes:
    a. base: High-level orchestrator
    * Delegates all specific operations to specialized classes
    * Maintains overall processing state
    * Coordinates processing pipeline

    b. parameters: Configuration management
    * Immutable after initialization
    * Validation logic
    * Default configurations  

    c. estimates: Results container
    * Stores processing results
    * Handles serialization/deserialization
    * Provides data access interfaces

2. Component Management:
    a. component: Abstract base class
    * Define interface for all component types
    * Common component operations  

    b. `SpatialComponent`, `TemporalComponent`, `BackgroundComponent`
    * Specific implementations for different component types
    * Encapsulated update logic
    * Quality metrics

3. Processing Classes:
    a. motion: 
    * Use implementations in `video_stabilization`

    b. initialization: 
    * Strategy pattern for different initialization methods
    * Bare initialization
    * Seeded initialization  

    c. update: 
    * HALS implementation
    * Component refinement
    * Update strategies

    d. deconvolution: 
    * Spike deconvolution
    * Different deconvolution strategies

4. Detection Classes:
    a. candidate: 
    * New component detection logic
    * CNN-based detection?? (idk seems kinda weird)
    * Statistical testing

    b. quality: 
    * Component evaluation
    * SNR calculation (how to address overlap with seed filters)
    * Correlation metrics

    c. merger: 
    * Overlap detection
    * Merging logic
    * Post-merge cleanup

5. Utility Classes:
    a. buffer: 
    * Efficient circular buffer
    * Frame management
    * Memory optimization

    b. visualization: 
    * Frame creation
    * Component visualization
    * Progress monitoring

    c. matrix_ops: 
    * Matrix operations
    * Efficient memory management
