#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>
#include <boost/thread.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
class DataLayer<Dtype>::sync {
  public:
   mutable boost::mutex master_to_worker_mutex_;
   mutable boost::mutex counter_mutex_;
};

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param), sync_(new sync()) {
	num_threads_ = param.num_threads();
    worker_data_full_ = false;
        LOG(INFO) << "Starting " << num_threads_ << " worker threads";
	for (int k = 0; k < num_threads_; k++) {
	  workers_.push_back(new DataLayerWorker(this, param));
	  workers_[k]->StartInternalThread();
	}
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
	for (int k = 0; k < num_threads_; k++) {
	  workers_[k]->StopInternalThread();
	}
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // set up parallel
  done_count_ = 0;
  worker_data_full_ = false;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    //printf("Master on item_id %d\n", item_id);
    //timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    //read_time += timer.MicroSeconds();
    //timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
	Dtype* ptr = top_data + offset;
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }

    // pass datum and prt to worker
    //printf("Master waiting for worker to take Datum\n");
    while(worker_data_full_) 
      boost::this_thread::yield();
    //printf("Master done waiting for worker to take Datum\n");
    sync_->master_to_worker_mutex_.lock();
    //printf("Master obtained lock\n");

    master_to_worker_datum_ = &datum;
    master_to_worker_ptr_ = ptr;
    worker_data_full_ = true;
    sync_->master_to_worker_mutex_.unlock();
    //printf("Master released lock\n");

    // worker does this
    //this->data_transformer_->Transform(datum, &(this->transformed_data_));
    //reader_.free().push(const_cast<Datum*>(&datum));

    //this->transformed_data_.set_cpu_data(top_data + offset);
    // Copy label.
    //trans_time += timer.MicroSeconds();

  }

  printf("Master done with batch\n");
  while (done_count_ < batch_size){ 
      boost::this_thread::yield(); // Wait for all workers to finish 
  }
  printf("Master has detected all workers are done\n");
  //timer.Stop();
  batch_timer.Stop();
  LOG(INFO) << "Prefetch batch: " << batch_timer.MicroSeconds() / 1000 << " ms.";
  LOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
DataLayer<Dtype>::DataLayerWorker::DataLayerWorker(DataLayer<Dtype>* parent, const LayerParameter& param): 
		transform_param_(param.transform_param()), parent_(parent) {
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, parent_->phase_));
}


template<typename Dtype>
void DataLayer<Dtype>::DataLayerWorker::InternalThreadEntry() {
  Dtype* ptr;
  bool got_data; 
  Datum* datum = NULL;
  printf("Worker started\n");
  while (!must_stop()) {
    got_data = false;
    parent_->sync_->master_to_worker_mutex_.lock();
    if (parent_->worker_data_full_) {
      //printf("-----Worker got work!(%d)\n", parent_->done_count_);
      datum = parent_->master_to_worker_datum_;
      ptr = parent_->master_to_worker_ptr_;
      parent_->worker_data_full_ = false;
      got_data = true;
    }	
    parent_->sync_->master_to_worker_mutex_.unlock();

    if (got_data) {
      this->data_transformer_->Transform(*datum, ptr);
      parent_->reader_.free().push(const_cast<Datum*>(datum));
      parent_->sync_->counter_mutex_.lock();
      parent_->done_count_++;
      //printf("-----Worker done, parent->DoneCount:%d\n", parent_->done_count_);
      parent_->sync_->counter_mutex_.unlock();
    } else {
      boost::this_thread::yield();
    }
  }
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
