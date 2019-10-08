#pragma once
#include <Windows.h>
#include <atlstr.h>
#define BUFFER_SIZE 128

class SerialPort
{
public:
	SerialPort();
	~SerialPort();
private:
	HANDLE  m_hComm;
	DCB     m_dcb;
	COMMTIMEOUTS m_CommTimeouts;
	bool    m_bPortReady;
	bool    m_bWriteRC;
	bool    m_bReadRC;
	DWORD   m_iBytesWritten;
	DWORD   m_iBytesRead;
	DWORD   m_dwBytesRead;

public:
	void ClosePort();
	bool readResponse(char &resp);
	//send pidValue Command
	bool runCommand(byte * cmd, char * data, unsigned int dataLength);
	//send AT Command
	bool runCommand(const char *cmd, char *data, unsigned int dataLength);
	void getBytes(byte *cmd, byte *values, unsigned int numValue);
	bool OpenPort(CString portname);
	bool SetCommunicationTimeouts(DWORD ReadIntervalTimeout,
		DWORD ReadTotalTimeoutMultiplier, DWORD ReadTotalTimeoutConstant,
		DWORD WriteTotalTimeoutMultiplier, DWORD WriteTotalTimeoutConstant);
	bool ConfigurePort(DWORD BaudRate, BYTE ByteSize, DWORD fParity,
		BYTE  Parity, BYTE StopBits);
};

