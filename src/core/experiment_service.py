
from typing import Any, Dict
from uuid import UUID

from .database.repositories.protocols import ExperimentRepository 
from ..shared.dtos.experiments import CreateExperimentDto, ExperimentDto, PaginationExperimentsDto, PaginatedResponseDto

import logging 

class ExperimentService: 
    def __init__(self):
        self.experiment_repo = ExperimentRepository() 
        self.logger = logging.getLogger(__name__) 

    async def add_experiment_async(
        self,
        run_name: str, 
        model_id: UUID,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        model_file_path: str | None
    ) -> UUID: 
        experiment = await self.experiment_repo.log_experiment(model_id=model_id, 
                                                               name=run_name, 
                                                               metrics=metrics,
                                                               params=params,
                                                               model_file_path=model_file_path)

        if experiment is None: 
            raise Exception("Experiment is not created") 

        return experiment 

    async def get_all_experiments_async(
        self,
        pagination: PaginationExperimentsDto
    ) -> PaginatedResponseDto: 
        
        self.logger.info(f"Fetching experiments: page={pagination.page}, "
                        f"page_size={pagination.page_size}, "
                        f"model_id={pagination.model_id}") 

        offset = (pagination -1 ) * pagination.page_size 

        experiments_data = await self.experiment_repo.get_all_experiments(
            limit=pagination.page_size,
            offset=offset,
            model_id=pagination.model_id
        )

        experiments = [
            ExperimentDto(
                id=exp["id"],
                model_id=exp["model_id"],
                run_name=exp["run_name"],
                params=exp["params"],
                metrics=exp["metrics"],
                model_file_path=exp.get("model_file_path")
            )
            for exp in experiments_data
        ]
        
        total = await self.experiment_repo.get_experiments_count(
            model_id=pagination.model_id
        )
        
        total_pages = (total + pagination.page_size - 1) // pagination.page_size if total > 0 else 0
        
        return PaginatedResponseDto(
            items=experiments,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_previous=pagination.page > 1
        )