#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

struct EdgeDetection {
	void operator()(const cv::Mat& input, cv::Mat& output) const {
		cv::Canny(input, output, 50, 150);
	}
};

struct Blur {
	void operator()(const cv::Mat& input, cv::Mat& output) const {
		cv::GaussianBlur(input, output, cv::Size(7, 7), 0);
	}
};

void apply_filters_sequential(const cv::Mat& input, cv::Mat& edges_output, cv::Mat& blurred_output) {
	EdgeDetection edge_detection;
	Blur blur;
	edge_detection(input, edges_output);
	blur(input, blurred_output);
}

void apply_filters_task_parallelism(const cv::Mat& input, cv::Mat& edges_output, cv::Mat& blurred_output) {
	EdgeDetection edge_detection;
	Blur blur;

	std::thread edge_detection_thread(edge_detection, std::cref(input), std::ref(edges_output));
	std::thread blur_thread(blur, std::cref(input), std::ref(blurred_output));

	edge_detection_thread.join();
	blur_thread.join();
}

void apply_filters_data_parallelism(const cv::Mat& input, cv::Mat& output, int start_row, int end_row) {
	cv::Mat segment = input.rowRange(start_row, end_row).clone();

	cv::Mat edges_output, blurred_output;
	cv::Canny(segment, edges_output, 50, 150);
	cv::GaussianBlur(segment, blurred_output, cv::Size(7, 7), 0);

	edges_output.copyTo(output.rowRange(start_row, end_row));
}

int main() {
	std::string input_image_path = "C:\\Main\\College\\parallel\\project\\img\\Eren.jpg";
	cv::Mat input = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
	if (input.empty()) {
		std::cerr << "Failed to open image file: " << input_image_path << std::endl;
		return -1;
	}

	cv::Mat edges_output = input.clone();
	cv::Mat blurred_output = input.clone();

	std::cout << "Choose parallelism method:\n1. Task parallelism\n2. Data parallelism\n3. Both\n";
	int choice;
	std::cin >> choice;

	auto start = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff;

	if (choice == 1) {
		start = std::chrono::high_resolution_clock::now();
		apply_filters_task_parallelism(input, edges_output, blurred_output);
		end = std::chrono::high_resolution_clock::now();
		diff = end - start;
		std::cout << "Task parallelism time: " << diff.count() << " s\n";
	}
	else if (choice == 2) {
		int num_threads = std::thread::hardware_concurrency();
		int rows_per_thread = input.rows / num_threads;

		std::vector<std::thread> threads;

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < num_threads; ++i) {
			int start_row = i * rows_per_thread;
			int end_row = (i == num_threads - 1) ? input.rows : start_row + rows_per_thread;
			threads.push_back(std::thread(apply_filters_data_parallelism, std::cref(input), std::ref(edges_output), start_row, end_row));
			threads.push_back(std::thread(apply_filters_data_parallelism, std::cref(input), std::ref(blurred_output), start_row, end_row));
		}

		for (auto& thread : threads) {
			thread.join();
		}
		end = std::chrono::high_resolution_clock::now();
		diff = end - start;
		std::cout << "Data parallelism time: " << diff.count() << " s\n";
	}
	else if (choice == 3) {
		// Task parallelism
		start = std::chrono::high_resolution_clock::now();
		apply_filters_task_parallelism(input, edges_output, blurred_output);
		end = std::chrono::high_resolution_clock::now();
		diff = end - start;
		std::cout << "Task parallelism time: " << diff.count() << " s\n";

		// Data parallelism
		int num_threads = std::thread::hardware_concurrency();
		int rows_per_thread = input.rows / num_threads;

		std::vector<std::thread> threads;

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < num_threads; ++i) {
			int start_row = i * rows_per_thread;
			int end_row = (i == num_threads - 1) ? input.rows : start_row + rows_per_thread;

			threads.push_back(std::thread(apply_filters_data_parallelism, std::cref(input), std::ref(edges_output), start_row, end_row));
			threads.push_back(std::thread(apply_filters_data_parallelism, std::cref(input), std::ref(blurred_output), start_row, end_row));
		}

		for (auto& thread : threads) {
			thread.join();
		}
		end = std::chrono::high_resolution_clock::now();
		diff = end - start;
		std::cout << "Data parallelism time: " << diff.count() << " s\n";
	}
	else {
		std::cerr << "Invalid choice\n";
		return -1;
	}

	// Sequential
	start = std::chrono::high_resolution_clock::now();
	apply_filters_sequential(input, edges_output, blurred_output);
	end = std::chrono::high_resolution_clock::now();
	diff = end - start;
	std::cout << "Sequential time: " << diff.count() << " s\n";

	std::string output_edges_path = "C:\\Main\\College\\parallel\\project\\img\\edge.jpg";
	cv::imwrite(output_edges_path, edges_output);

	std::string output_blurred_path = "C:\\Main\\College\\parallel\\project\\img\\blur.jpg";
	cv::imwrite(output_blurred_path, blurred_output);

	return 0;
}
