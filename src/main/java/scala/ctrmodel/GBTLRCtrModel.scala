package scala.ctrmodel

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.gbtlr.{GBTLRClassificationModel, GBTLRClassifier}
import org.apache.spark.sql.DataFrame

class GBTLRCtrModel {

  var _pipelineModel:PipelineModel = _
  var _model:GBTLRClassificationModel = _

  def train(samples:DataFrame) : Unit = {

    _model = new GBTLRClassifier()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setGBTMaxIter(10)
      .setLRMaxIter(100)
      .setRegParam(0.01)
      .setElasticNetParam(0.5)
      .fit(samples)
  }

  def transform(samples:DataFrame):DataFrame = {
    _model.transform(samples)
  }
}
