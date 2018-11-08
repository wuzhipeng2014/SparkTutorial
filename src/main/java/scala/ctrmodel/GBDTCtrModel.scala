package scala.ctrmodel

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.sql.DataFrame

class GBDTCtrModel {

  var _pipelineModel:PipelineModel = _
  var _model:GBTClassificationModel = _

  def train(samples:DataFrame) : Unit = {

    _model = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .fit(samples)
  }

  def transform(samples:DataFrame):DataFrame = {
    _model.transform(samples)
  }
}
