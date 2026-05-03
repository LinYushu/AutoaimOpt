#include "threads/pipeline.h"
#include <unistd.h>
#include <iostream>
#include <thread>
#include <openrm/cudatools.h>
#include "nvtx3/nvtx3.hpp"

using namespace rm;
using namespace nvinfer1;
using namespace nvonnxparser;

void Pipeline::detector_baseline_thread(
    std::mutex& mutex_in, bool& flag_in, std::shared_ptr<rm::Frame>& frame_in, 
    std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out
) {
    auto param = Param::get_instance();
    auto garage = Garage::get_instance();

    std::string yolo_type = (*param)["Model"]["YoloArmor"]["Type"];

    int    infer_width       = (*param)["Model"]["YoloArmor"][yolo_type]["InferWidth"];
    int    infer_height      = (*param)["Model"]["YoloArmor"][yolo_type]["InferHeight"];
    int    class_num         = (*param)["Model"]["YoloArmor"][yolo_type]["ClassNum"];
    int    locate_num        = (*param)["Model"]["YoloArmor"][yolo_type]["LocateNum"];
    int    color_num         = (*param)["Model"]["YoloArmor"][yolo_type]["ColorNum"];
    int    bboxes_num        = (*param)["Model"]["YoloArmor"][yolo_type]["BboxesNum"];
    double confidence_thresh = (*param)["Model"]["YoloArmor"][yolo_type]["ConfThresh"];
    double nms_thresh        = (*param)["Model"]["YoloArmor"][yolo_type]["NMSThresh"];
    std::vector<int> class_map = (*param)["Model"]["YoloArmor"][yolo_type]["ClassMap"].get<std::vector<int>>();
    std::vector<int> color_map = (*param)["Model"]["YoloArmor"][yolo_type]["ColorMap"].get<std::vector<int>>();

    int tensor_locate_dim = locate_num;
    int tensor_color_dim = color_num;
    int tensor_class_dim = class_num;

    if (yolo_type == "V5") {
        tensor_locate_dim = 4;
        tensor_color_dim = 0;  // 强制修正：V5模型显存中无颜色输出，防止错读！
    } else if (yolo_type == "FP") {
        tensor_locate_dim = 8;
        tensor_color_dim = 0;  // 强制修正：FP模型显存中无颜色输出！
    } else if (yolo_type == "FPX") {
        tensor_locate_dim = 8;
        tensor_color_dim = 4;  // 强制修正：FPX 显存中分配了4位给颜色！
    }

    size_t yolo_struct_size = sizeof(float) * static_cast<size_t>(locate_num + 1 + color_num + class_num);

    size_t input_step_floats = infer_width * infer_height * 3;
    // output_step_floats 专门用于 TensorRT 输出在 Device 上的偏移（保留原逻辑）
    size_t output_step_bytes = (yolo_struct_size * bboxes_num + sizeof(float) + 255) & ~255;
    size_t output_step_floats = output_step_bytes / sizeof(float);
    int local_buf_idx = 0;

    int max_output_num = 100; // NMS 候选集上限
    // compact_step_bytes 专门用于精简数据传回 Host 时的内存偏移
    size_t compact_step_bytes = sizeof(int) + max_output_num * sizeof(YoloDetectionRaw);

    // NMS 专用的 Device 缓冲区 (Graph 内使用)
    YoloDetectionRaw* d_nms_output;
    int* d_nms_count;
    cudaMalloc(&d_nms_output, max_output_num * sizeof(YoloDetectionRaw));
    cudaMalloc(&d_nms_count, sizeof(int));

    cudaGraph_t graph[2];
    cudaGraphExec_t graphExec[2];
    bool graph_created[2] = {false, false};

    std::mutex mutex;
    TimePoint tp0, tp1, tp2;
    while (true) {
        if (!Data::armor_mode) {
            std::unique_lock<std::mutex> lock(mutex);
            armor_cv_.wait(lock, [this]{return Data::armor_mode;});
        }

        tp0 = getTime();

        std::unique_lock<std::mutex> lock_in(mutex_in);
        if (!flag_in) {
            detect_cv_.wait_for(lock_in, std::chrono::milliseconds(15), [&] {
                return flag_in;
            });
        } 
        if (!flag_in) {
            lock_in.unlock();
            continue;
        }
        std::shared_ptr<rm::Frame> frame = frame_in;
        flag_in = false;
        lock_in.unlock();


        // double wait_time_ms = getDoubleOfS(tp0, getTime()) * 1000.0;
        // // 150帧时理论每帧间隔不到7ms，如果等待超过 12ms，说明上游(相机或预处理)卡了
        // if (wait_time_ms > 5.0) { 
        //     // rm::message("[AbnormalSpike-Probe 1] Det waiting for image: " + std::to_string(wait_time_ms) + " ms");
        // }

        tp1 = getTime();
        {
        nvtx3::scoped_range marker("Det");
        
        // 分离指针：Device 端按照庞大步长偏移
        float *curr_input_dev = armor_input_device_buffer_ + (local_buf_idx * input_step_floats);
        float* curr_output_dev = armor_output_device_buffer_ + (local_buf_idx * output_step_floats);
        
        // 分离指针：Host 端按照极小步长偏移，按 char* 计算字节数
        void* curr_output_host = (char*)armor_output_host_buffer_ + (local_buf_idx * compact_step_bytes);

        cudaStreamWaitEvent(detect_stream_, resize_complete_event_[local_buf_idx], 0);

        if (!graph_created[local_buf_idx]) {
            cudaStreamBeginCapture(detect_stream_, cudaStreamCaptureModeGlobal);

            // 节点 1：TensorRT 网络推理
            detectEnqueue(
                curr_input_dev, curr_output_dev, &armor_context_, &detect_stream_
            );
            
            // 计算坐标还原参数
            float width_ratio = (float)frame_in->width / (float)infer_width;
            float height_ratio = (float)frame_in->height / (float)infer_height;
            float ratio = (width_ratio > height_ratio) ? width_ratio : height_ratio;
            float top_move = (width_ratio > height_ratio) ? ((float)infer_height * ratio - frame_in->height) / 2.f : 0.f;
            float left_move = (width_ratio > height_ratio) ? 0.f : ((float)infer_width * ratio - frame_in->width) / 2.f;

            // 节点 2：启动自定义 CUDA NMS 核函数
            rm::launch_yolo_decode_and_nms(
                curr_output_dev, d_nms_output, d_nms_count, bboxes_num,
                tensor_locate_dim, tensor_color_dim, tensor_class_dim, // 传入真实的物理维度
                infer_width, infer_height,                             // 传入分辨率以供边缘过滤
                confidence_thresh, nms_thresh,
                ratio, left_move, top_move, max_output_num, detect_stream_);

            // 节点 3：极小量数据拷贝回 CPU (共计约 6.4 KB)
            cudaMemcpyAsync(curr_output_host, d_nms_count, sizeof(int), cudaMemcpyDeviceToHost, detect_stream_);
            cudaMemcpyAsync((char*)curr_output_host + sizeof(int), d_nms_output, max_output_num * sizeof(YoloDetectionRaw), cudaMemcpyDeviceToHost, detect_stream_);
            
            cudaError_t end_err = cudaStreamEndCapture(detect_stream_, &graph[local_buf_idx]);
            if (end_err != cudaSuccess) {
                rm::message("Graph Capture Failed", rm::MSG_ERROR);
                exit(-1);
            }
            
            cudaError_t inst_err = cudaGraphInstantiate(&graphExec[local_buf_idx], graph[local_buf_idx], NULL, NULL, 0);
            if (inst_err != cudaSuccess) {
                rm::message("Graph Instantiate Failed", rm::MSG_ERROR);
                exit(-1);
            }
            graph_created[local_buf_idx] = true;
            rm::message("CUDA Graph created for buffer " + std::to_string(local_buf_idx), rm::MSG_OK);
        }
        
        // 发射打包好的图！(这只要 1 次 API 调用)
        cudaGraphLaunch(graphExec[local_buf_idx], detect_stream_);

        cudaEventRecord(detect_complete_event_[local_buf_idx], detect_stream_);

        // TimePoint gpu_wait_start = getTime();
        cudaEventSynchronize(detect_complete_event_[local_buf_idx]);
        // double gpu_sync_time_ms = getDoubleOfS(gpu_wait_start, getTime()) * 1000.0;
        // // 正常的网络推理+NMS在边缘端显卡上通常在 2~5ms 以内
        // if (gpu_sync_time_ms > 6.0) {
        //     // rm::message("[AbnormalSpike-Probe 2] GPU infer and sync time exception:" + std::to_string(gpu_sync_time_ms) + " ms");
        // }

        frame->yolo_list.clear();
        int valid_count = *(int*)curr_output_host;
        valid_count = std::min(valid_count, max_output_num);
        YoloDetectionRaw* raw_results = (YoloDetectionRaw*)((char*)curr_output_host + sizeof(int));

        for (int i = 0; i < valid_count; ++i) {
            if (!raw_results[i].keep) continue;

            YoloRect rect;
            rect.box = cv::Rect(raw_results[i].box[0] - raw_results[i].box[2]/2, 
                                raw_results[i].box[1] - raw_results[i].box[3]/2, 
                                raw_results[i].box[2], raw_results[i].box[3]);
            rect.confidence = raw_results[i].confidence;
            
            // 使用 ClassMap 纠正输出类别
            int raw_class = raw_results[i].class_id;
            rect.class_id = (!class_map.empty() && raw_class < class_map.size()) ? class_map[raw_class] : raw_class;
            
            // 如果显存物理维度上无颜色输出(如V5/FP)，则从 ColorMap 映射；否则直接读取
            if (tensor_color_dim == 0 && !color_map.empty() && raw_class < color_map.size()) {
                rect.color_id = color_map[raw_class];
            } else {
                rect.color_id = raw_results[i].color_id;
            }
            
            // Tracker 四点顺序还原：TL, BL, BR, TR
            int x_index[4] = {0, 6, 2, 4}; 
            int y_index[4] = {1, 7, 3, 5};
            for (int k = 0; k < 4; ++k) {
                rect.four_points.push_back(cv::Point2f(
                    raw_results[i].pose[x_index[k]], 
                    raw_results[i].pose[y_index[k]]
                ));
            }
            frame->yolo_list.push_back(rect);
        }

        local_buf_idx = 1 - local_buf_idx;

        if (frame->yolo_list.empty()) {
            if (Data::image_flag) imshow(frame);
            continue;
        }
        } // nvtx scope 结束

        tp2 = getTime();

        // double process_time_ms = getDoubleOfS(tp1, tp2) * 1000.0;
        // // 正常情况下纯检测处理(不含等图)耗时非常稳定，若超过 10ms 则说明 CPU/GPU 配合出现严重阻塞
        // if (process_time_ms > 7.0) {
        //     // rm::message("[AbnormalSpike-Probe 3] Total time of single frame detection:" + std::to_string(process_time_ms) + " ms");
        // }

        if (Data::pipeline_delay_flag) {
          rm::message("detect", getDoubleOfS(tp1, tp2) * 1000);
        }

        std::unique_lock<std::mutex> lock_out(mutex_out);
        frame_out = frame;
        flag_out = true;
        lock_out.unlock();
        tracker_in_cv_.notify_one();
    }
}