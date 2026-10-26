#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "memory/nvme_loader.hpp"
#include <windows.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>

static std::wstring get_temp_path() {
    wchar_t buf[MAX_PATH];
    DWORD len = GetTempPathW(MAX_PATH, buf);
    if (len == 0 || len > MAX_PATH) return L"C:\a11_test_tensor.bin";
    return std::wstring(buf);
}

static std::wstring create_dummy_file(const std::wstring& path, size_t byte_count) {
    const size_t sector_size = 65536;
    size_t aligned_size = ((byte_count + sector_size - 1) / sector_size) * sector_size;
    HANDLE hFile = CreateFileW(path.c_str(), GENERIC_WRITE | GENERIC_READ, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(hFile != INVALID_HANDLE_VALUE);
    std::vector<uint8_t> buffer(aligned_size);
    for (size_t i = 0; i < aligned_size; ++i) { buffer[i] = static_cast<uint8_t>((i & 0xFF) ^ 0x5A); }
    DWORD written = 0;
    BOOL result = WriteFile(hFile, buffer.data(), static_cast<DWORD>(aligned_size), &written, nullptr);
    REQUIRE(result == TRUE);
    CloseHandle(hFile);
    return path;
}

static void cleanup_file(const std::wstring& path) { DeleteFileW(path.c_str()); }

TEST_CASE("NvmeAsyncLoader instantiation and error handling", "[nvme][io]") {
    A11::Memory::NvmeAsyncLoader loader;
    REQUIRE(!loader.is_open());
    bool opened = loader.open_file(L"C:\nonexistent_path_for_test\tensor.bin");
    REQUIRE(!opened);
    REQUIRE(!loader.is_open());

    A11::Memory::NvmeAsyncLoader loader2;
    loader2 = std::move(loader);
    REQUIRE(!loader2.is_open());
}

TEST_CASE("NvmeAsyncLoader async read with pinned buffer", "[nvme][io][async]") {
    const size_t file_size = 10 * 1024 * 1024;
    std::wstring temp_dir = get_temp_path();
    std::wstring dummy_path = temp_dir + L"a11_test_tensor.bin";
    cleanup_file(dummy_path);

    SECTION("Create dummy tensor file and async read") {
        create_dummy_file(dummy_path, file_size);
        A11::Memory::NvmeAsyncLoader loader;
        REQUIRE(loader.open_file(dummy_path));
        REQUIRE(loader.is_open());

        const size_t chunk_size = 1024 * 1024;
        const size_t num_chunks = file_size / chunk_size;
        void* host_pinned_buffer = VirtualAlloc(nullptr, chunk_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        REQUIRE(host_pinned_buffer != nullptr);

        auto start_time = std::chrono::high_resolution_clock::now();
        size_t total_bytes_read = 0;
        bool all_passed = true;

        for (size_t i = 0; i < num_chunks && all_passed; ++i) {
            uint64_t offset = static_cast<uint64_t>(i * chunk_size);
            OVERLAPPED overlapped{};
            ZeroMemory(&overlapped, sizeof(OVERLAPPED));

            bool initiated = loader.read_tensor_async(offset, chunk_size, host_pinned_buffer, &overlapped);
            REQUIRE(initiated);

            DWORD bytes_transferred = 0;
            bool completed = loader.wait_for_completion(&overlapped, &bytes_transferred, 30000);
            REQUIRE(completed);
            REQUIRE(bytes_transferred == chunk_size);
            total_bytes_read += bytes_transferred;

            const uint8_t* buf = static_cast<const uint8_t*>(host_pinned_buffer);
            for (size_t j = 0; j < chunk_size && all_passed; ++j) {
                uint64_t global_offset = offset + j;
                uint8_t expected = static_cast<uint8_t>((global_offset & 0xFF) ^ 0x5A);
                if (buf[j] != expected) {
                    std::cerr << "Checksum mismatch at chunk=" << i << " off=" << j
                              << " exp=0x" << std::hex << (int)expected << " got=0x" << (int)buf[j] << std::dec << std::endl;
                    all_passed = false;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double throughput_mbps = (static_cast<double>(total_bytes_read) / (1024.0 * 1024.0)) / ((double)duration_ms / 1000.0);

        REQUIRE(total_bytes_read == file_size);
        REQUIRE(all_passed);

        std::ostringstream oss;
        oss << "Async NVMe throughput: " << std::fixed << std::setprecision(2)
            << throughput_mbps << " MB/sec (" << total_bytes_read / (1024*1024) << " MB in "
            << duration_ms << " ms, " << num_chunks << " chunks)" << std::endl;
        std::cerr << oss.str();

        loader.close_file();
        REQUIRE(!loader.is_open());
        VirtualFree(host_pinned_buffer, 0, MEM_RELEASE);
    }

    cleanup_file(dummy_path);
}
