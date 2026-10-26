#pragma once

#ifndef A11_NVME_LOADER_HPP
#define A11_NVME_LOADER_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <windows.h>

namespace A11 {
namespace Memory {

class NvmeAsyncLoader {
public:
    NvmeAsyncLoader();
    ~NvmeAsyncLoader();

    NvmeAsyncLoader(const NvmeAsyncLoader&) = delete;
    NvmeAsyncLoader& operator=(const NvmeAsyncLoader&) = delete;
    NvmeAsyncLoader(NvmeAsyncLoader&& other) noexcept;
    NvmeAsyncLoader& operator=(NvmeAsyncLoader&& other) noexcept;

    bool open_file(const std::wstring& file_path);
    bool read_tensor_async(uint64_t offset, size_t bytes, void* host_pinned_buffer, OVERLAPPED* overlapped_struct);
    bool wait_for_completion(OVERLAPPED* overlapped_struct, DWORD* bytes_transferred, DWORD timeout_ms = 5000);
    void close_file();
    bool is_open() const;

private:
    HANDLE file_handle_ = INVALID_HANDLE_VALUE;
    static void log_win32_error(const char* operation, DWORD error_code);
};

} // namespace Memory
} // namespace A11

#endif // A11_NVME_LOADER_HPP
