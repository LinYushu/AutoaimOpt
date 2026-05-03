#include "threads/pipeline.h"
#include <unistd.h>
#include <iostream>
#include <thread>
#include <openrm/cudatools.h>
#include "nvtx3/nvtx3.hpp"

using namespace rm;
using namespace nvinfer1;
using namespace nvonnxparser;

void Pipeline::preprocessor_baseline_thread(
    std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out
) {
    auto param = Param::get_instance();
    auto garage = Garage::get_instance();

    std::string yolo_type   = (*param)["Model"]["YoloArmor"]["Type"];
    std::string onnx_file   = (*param)["Model"]["YoloArmor"][yolo_type]["DirONNX"];
    std::string engine_file = (*param)["Model"]["YoloArmor"][yolo_type]["DirEngine"];

    int  infer_width     = (*param)["Model"]["YoloArmor"][yolo_type]["InferWidth"];
    int  infer_height    = (*param)["Model"]["YoloArmor"][yolo_type]["InferHeight"];
    int  class_num       = (*param)["Model"]["YoloArmor"][yolo_type]["ClassNum"];
    int  locate_num      = (*param)["Model"]["YoloArmor"][yolo_type]["LocateNum"];
    int  color_num       = (*param)["Model"]["YoloArmor"][yolo_type]["ColorNum"];
    int  bboxes_num      = (*param)["Model"]["YoloArmor"][yolo_type]["BboxesNum"];
    bool hist_input_flag = (*param)["Model"]["YoloArmor"][yolo_type]["NeedHist"];

    if (access(engine_file.c_str(), F_OK) == 0) {
        if (!rm::initTrtEngine(engine_file, &armor_context_)) exit(-1);
    } else if (access(onnx_file.c_str(), F_OK) == 0){
        if (!rm::initTrtOnnx(onnx_file, engine_file, &armor_context_, 1U)) exit(-1);
    } else {
        rm::message("No model file found!", rm::MSG_ERROR);
        exit(-1);
    }

    size_t yolo_struct_size = sizeof(float) * static_cast<size_t>(locate_num + 1 + color_num + class_num);
    mallocYoloDetectBuffer(
        &armor_input_device_buffer_, 
        &armor_output_device_buffer_, 
        &armor_output_host_buffer_, 
        infer_width, 
        infer_height, 
        yolo_struct_size,
        bboxes_num,
        2
    );

    // 计算偏移量 (Offset)
    size_t input_step_floats = infer_width * infer_height * 3; // float数量
    size_t output_step_bytes = (yolo_struct_size * bboxes_num + sizeof(float) + 255) & ~255;
    size_t output_step_floats = output_step_bytes / sizeof(float);

    buffer_idx_ = 0;
    first_run_ = true;

    std::mutex mutex;
    TimePoint frame_wait, flag_wait;
    TimePoint tp0, tp1, tp2;
    while(true) {
        // rm::message("Pre " + std::to_string(Data::enemy_color));
        if (!Data::armor_mode) {
            std::unique_lock<std::mutex> lock(mutex);
            armor_cv_.wait(lock, [this]{return Data::armor_mode;});
        }

        Camera* camera = Data::camera[Data::camera_index];
        std::shared_ptr<rm::Frame> frame = camera->buffer->pop();

        if (frame == nullptr) {
            std::unique_lock<std::mutex> lock(camera->frame_mutex);
            camera->frame_cv.wait_for(lock, std::chrono::milliseconds(15), [&] {
                frame = camera->buffer->pop();
                return frame != nullptr;
            });
            if (frame == nullptr) continue;
        }

        tp1 = getTime();

        {
        nvtx3::scoped_range marker("Pre");
        if (!first_run_) {
            cudaStreamWaitEvent(resize_stream_, detect_complete_event_[buffer_idx_], 0);
        }

        memcpyYoloCameraBuffer(
            frame->image->data,
            camera->rgb_host_buffer,
            camera->rgb_device_buffer,
            frame->width,
            frame->height,
            &resize_stream_
        );
        
        // 计算当前帧的显存指针偏移
        float* curr_input_dev = armor_input_device_buffer_ + (buffer_idx_ * input_step_floats);
        
        resize(
            camera->rgb_device_buffer,
            frame->width,
            frame->height,
            curr_input_dev,
            infer_width,
            infer_height,
            (void*)resize_stream_
        );
        cudaEventRecord(resize_complete_event_[buffer_idx_], resize_stream_);

        if (Data::record_mode) { record(frame); }
        }

        tp2 = getTime();
        if (Data::pipeline_delay_flag) {
          rm::message("preprocess", getDoubleOfS(tp1, tp2) * 1000);
        //   rm::message("Pre: " + std::to_string(getDoubleOfS(tp1, tp2) * 1000) + "ms");
        }

        buffer_idx_ = 1 - buffer_idx_;
        first_run_ = false;

        // std::unique_lock<std::mutex> lock_out(mutex_out);
        // frame_out = frame;
        // flag_out = true;

        std::unique_lock<std::mutex> lock_out(mutex_out);
        frame_out = frame;
        flag_out = true;
        lock_out.unlock();
        // 瞬间唤醒检测线程
        detect_cv_.notify_one();
    }
}