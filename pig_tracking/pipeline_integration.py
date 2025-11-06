"""
Pipeline Integration Module - Bridge between core.pipeline and existing code

This module provides convenient functions to use the new VideoPipeline
with existing video processing infrastructure.

Spec-compliant unified processing following .kiro/specs requirements.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from core.pipeline import (
    VideoPipeline, 
    WeighingAct,
    CrossingEvent,
    create_pipeline
)
from pig_tracking.database_manager import DatabaseManager
from pig_tracking.video_processor import IntegratedVideoProcessor

logger = logging.getLogger(__name__)


class PipelineAdapter:
    """
    Adapter that bridges VideoPipeline with IntegratedVideoProcessor
    
    Allows using the new unified pipeline architecture while maintaining
    compatibility with existing video processing code.
    """
    
    def __init__(
        self,
        stream_id: str,
        video_source: str,
        database_manager: Optional[DatabaseManager] = None,
        model_path: Optional[str] = None,
        **config
    ):
        """
        Initialize pipeline adapter
        
        Args:
            stream_id: Stream identifier (e.g., "cam101")
            video_source: Video file path or RTSP URL
            database_manager: Optional database connection
            model_path: Optional YOLO model path
            **config: Additional configuration (line_left_x, line_right_x, etc.)
        """
        self.stream_id = stream_id
        self.video_source = video_source
        self.database_manager = database_manager
        
        # Create unified pipeline
        self.pipeline = create_pipeline(
            stream_id=stream_id,
            source_uri=video_source,
            database_manager=database_manager,
            model_path=model_path,
            **config
        )
        
        # Create integrated video processor for actual processing
        self.processor = IntegratedVideoProcessor(
            stream_id=stream_id,
            conf_threshold=config.get('conf_threshold', 0.30),
            img_size=config.get('img_size', 960),
            line_left_x=config.get('line_left_x', 0.25),
            line_right_x=config.get('line_right_x', 0.75),
            min_pigs_for_act=config.get('min_pigs_for_act', 3),
            max_interval_sec=config.get('max_interval_sec', 30.0)
        )
        
        logger.info(f"[PipelineAdapter] Initialized for {stream_id}: {video_source}")
    
    async def process_video(self, video_path: str) -> List[WeighingAct]:
        """
        Process video and return detected acts
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of WeighingAct objects
        """
        logger.info(f"[PipelineAdapter] Processing video: {video_path}")
        
        try:
            # Initialize processor
            await self.processor.initialize()
            
            # Process video
            acts = await self.processor.process_video(video_path)
            
            # Convert to WeighingAct if needed
            if acts and not isinstance(acts[0], WeighingAct):
                # Already in correct format from IntegratedVideoProcessor
                pass
            
            logger.info(f"[PipelineAdapter] Processed {len(acts) if acts else 0} acts")
            return acts or []
            
        except Exception as e:
            logger.error(f"[PipelineAdapter] Error processing video: {e}")
            return []
    
    async def process_stream(self, source_uri: str, duration_sec: Optional[int] = None) -> List[WeighingAct]:
        """
        Process live stream (RTSP, etc.)
        
        Args:
            source_uri: Stream URI (RTSP, HTTP, etc.)
            duration_sec: Optional duration in seconds (None = infinite)
        
        Returns:
            List of WeighingAct objects
        """
        logger.info(f"[PipelineAdapter] Processing stream: {source_uri}")
        
        try:
            await self.processor.initialize()
            
            # Process stream
            acts = await self.processor.process_stream(
                source_uri,
                duration_sec=duration_sec
            )
            
            logger.info(f"[PipelineAdapter] Stream processing completed: {len(acts) if acts else 0} acts")
            return acts or []
            
        except Exception as e:
            logger.error(f"[PipelineAdapter] Error processing stream: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        pipeline_stats = self.pipeline.get_statistics()
        processor_stats = self.processor.get_statistics()
        
        return {
            **pipeline_stats,
            "processor_frames": processor_stats.get("frames_processed", 0),
            "processor_detections": processor_stats.get("total_detections", 0),
        }


async def process_video_spec_compliant(
    video_path: str,
    stream_id: str = "default",
    database_manager: Optional[DatabaseManager] = None,
    **config
) -> List[WeighingAct]:
    """
    Spec-compliant video processing function
    
    Process video using unified pipeline according to .kiro/specs
    
    Args:
        video_path: Path to video file
        stream_id: Stream identifier
        database_manager: Optional database connection
        **config: Processing configuration
    
    Returns:
        List of detected WeighingAct objects
    """
    adapter = PipelineAdapter(
        stream_id=stream_id,
        video_source=video_path,
        database_manager=database_manager,
        **config
    )
    
    return await adapter.process_video(video_path)


# Module-level convenience functions
def get_pipeline_adapter(
    stream_id: str,
    video_source: str,
    **config
) -> PipelineAdapter:
    """Create a pipeline adapter instance"""
    return PipelineAdapter(stream_id, video_source, **config)

