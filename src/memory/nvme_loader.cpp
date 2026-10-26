#include "memory/nvme_loader.hpp"
#include <iostream>

namespace A11 {
namespace Memory {

NvmeAsyncLoader::NvmeAsyncLoader() = default;
NvmeAsyncLoader::~NvmeAsyncLoader() { close_file(); }
NvmeAsyncLoader::NvmeAsyncLoader(NvmeAsyncLoader&& other) noexcept : file_handle_(other.file_handle_) { other.file_handle_ = INVALID_HANDLE_VALUE; }

NvmeAsyncLoader& NvmeAsyncLoader::operator=(NvmeAsyncLoader&& other) noexcept {
    if (this != &other) { close_file(); file_handle_ = other.file_handle_; other.file_handle_ = INVALID_HANDLE_VALUE; }
    return *this;
}

void NvmeAsyncLoader::log_win32_error(const char* operation, DWORD error_code) {
    LPSTR buf = nullptr;
    size_t sz = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buf), 0, nullptr);
    if (sz > 0 && buf) {
        while (sz > 0 && (buf[sz-1]=='\n' || buf[sz-1]=='\r' || buf[sz-1]==' ')) --sz;
        std::cerr << "[NVMeAsyncLoader ERROR] " << operation
                  << " failed (code: " << error_code << "): "
                  << std::string(buf, sz) << std::endl;
        LocalFree(buf);
    } else {
        std::cerr << "[NVMeAsyncLoader ERROR] " << operation
                  << " failed with unknown code: " << error_code << std::endl;
    }
}

bool NvmeAsyncLoader::open_file(const std::wstring& file_path) {
    close_file();
    file_handle_ = CreateFileW(
        file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) { log_win32_error("CreateFileW", GetLastError()); return false; }
    std::wcerr << L"[NVMeAsyncLoader] File opened: " << file_path << std::endl;
    return true;
}

bool NvmeAsyncLoader::read_tensor_async(uint64_t offset, size_t bytes, void* host_pinned_buffer, OVERLAPPED* overlapped_struct) {
    if (file_handle_ == INVALID_HANDLE_VALUE) { log_win32_error("ReadFile", ERROR_INVALID_HANDLE); return false; }
    if (!host_pinned_buffer || !overlapped_struct) { std::cerr << "[NVMeAsyncLoader ERROR] null buffer or OVERLAPPED" << std::endl; return false; }

    ZeroMemory(overlapped_struct, sizeof(OVERLAPPED));
    overlapped_struct->Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    overlapped_struct->OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFFULL);

    overlapped_struct->hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    if (!overlapped_struct->hEvent) { log_win32_error("CreateEvent", GetLastError()); return false; }

    BOOL result = ReadFile(file_handle_, host_pinned_buffer, static_cast<DWORD>(bytes), nullptr, overlapped_struct);
    if (!result) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) { log_win32_error("ReadFile", err); CloseHandle(overlapped_struct->hEvent); overlapped_struct->hEvent = nullptr; return false; }
    }
    std::wcerr << L"[NVMeAsyncLoader] Async read initiated: offset=" << offset << L" bytes=" << bytes << std::endl;
    return true;
}

bool NvmeAsyncLoader::wait_for_completion(OVERLAPPED* overlapped_struct, DWORD* bytes_transferred, DWORD timeout_ms) {
    if (!overlapped_struct || !overlapped_struct->hEvent) { std::cerr << "[NVMeAsyncLoader ERROR] invalid OVERLAPPED" << std::endl; return false; }

    DWORD wr = WaitForSingleObject(overlapped_struct->hEvent, timeout_ms);
    if (wr == WAIT_TIMEOUT) { std::wcerr << L"[NVMeAsyncLoader WARNING] Timeout after " << timeout_ms << L" ms" << std::endl; return false; }
    if (wr != WAIT_OBJECT_0) { log_win32_error("WaitForSingleObject", GetLastError()); CloseHandle(overlapped_struct->hEvent); overlapped_struct->hEvent = nullptr; return false; }

    if (!GetOverlappedResult(file_handle_, overlapped_struct, bytes_transferred, FALSE)) { log_win32_error("GetOverlappedResult", GetLastError()); CloseHandle(overlapped_struct->hEvent); overlapped_struct->hEvent = nullptr; return false; }

    CloseHandle(overlapped_struct->hEvent);
    overlapped_struct->hEvent = nullptr;
    std::wcerr << L"[NVMeAsyncLoader] Completed: " << *bytes_transferred << L" bytes" << std::endl;
    return true;
}

void NvmeAsyncLoader::close_file() { if (file_handle_ != INVALID_HANDLE_VALUE) { CloseHandle(file_handle_); file_handle_ = INVALID_HANDLE_VALUE; } }
bool NvmeAsyncLoader::is_open() const { return file_handle_ != INVALID_HANDLE_VALUE; }

} // namespace Memory
} // namespace A11
