from typing import Any
import tlc

class SuperGradientsDetectionMetricsCollector(tlc.MetricsCollector):
    def __init__(self, input_table: tlc.Table):
        self.table = input_table
        super().__init__(compute_aggregates=False)

    def compute_metrics(self, batch: Any, predictor_output: tlc.PredictorOutput):
        predictions = predictor_output.forward
        detections_predicted = []
        