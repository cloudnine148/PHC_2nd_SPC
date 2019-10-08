#include "DataCommunication.hpp"

DataCommunication::DataCommunication(char *port)
{
	char* comPort = port;
	if (!serialConnect(comPort))
	{
		std::cout << "Serial port connection failed!!" << std::endl;
		Sleep(3000);
		//exit(0);
	}
	else
		std::cout << "Serial port connection success!!" << std::endl;

	//ProtocolInfo Initialize
	prtInfo.sync = 0x00;
	prtInfo.crc = 0x00;
	memset(&prtInfo.data, 0x00, sizeof(prtInfo.data));
}


DataCommunication::~DataCommunication()
{
}

/*
*		@brief		���� ���۽� ���۵� �޼���
*/
void DataCommunication::stop_and_stop_signal_datafield_initialize()
{
	prtInfo.data.detection_cnt = 0xff;
	prtInfo.data.l_risk1 = 0xff;
	prtInfo.data.l_position1 = 0xff;
	prtInfo.data.distance1 = 0xff;
	prtInfo.data.l_direction1 = 0xff;
	prtInfo.data.riskrate1 = 0xff;
	prtInfo.data.l_risk2 = 0xff;
	prtInfo.data.l_position2 = 0xff;
	prtInfo.data.distance2 = 0xff;
	prtInfo.data.l_direction2 = 0xff;
	prtInfo.data.riskrate2 =  0xff;
	prtInfo.data.l_risk3 = 0xff;
	prtInfo.data.l_position3 = 0xff;
	prtInfo.data.distance3 = 0xff;
	prtInfo.data.l_direction3 = 0xff;
	prtInfo.data.riskrate3 = 0xff;
}
void DataCommunication::Command_Start()
{
	char rsvMsg[5];

	memset(&prtInfo.data, 0x00, sizeof(prtInfo.data));

	stop_and_stop_signal_datafield_initialize();
	
	prtInfo.sync = 0x80;
	prtInfo.data.day_night = 0xff;
	prtInfo.data.command = 0x01;
	//prtInfo.data.command = 0x01;		// 0x01 : ���� ���۽�  ����

	DATA_BITFIELD *dt = &prtInfo.data;
	unsigned char* dataField = (unsigned char*)dt;
	unsigned char crcData = crcCalculate8_SAE_J1850(dataField, 11);
	prtInfo.crc = crcData;
	PROTOCOL_INFO* ptr = &prtInfo;

	byte* cmd = (byte*)ptr;
	serial.runCommand(cmd, rsvMsg, sizeof(prtInfo));
}

/*
*		@brief		����� ���۵� �޼���
*/
void DataCommunication::Command_Stop()
{
	char rsvMsg[5];

	memset(&prtInfo.data, 0x00, sizeof(prtInfo.data));

	stop_and_stop_signal_datafield_initialize();

	prtInfo.sync = 0x80;
	prtInfo.data.day_night = 0xff;
	prtInfo.data.command = 0x02; // ����� ����

	DATA_BITFIELD *dt = &prtInfo.data;
	unsigned char* dataField = (unsigned char*)dt;
	unsigned char crcData = crcCalculate8_SAE_J1850(dataField, 11);
	prtInfo.crc = crcData;
	PROTOCOL_INFO* ptr = &prtInfo;

	byte* cmd = (byte*)ptr;
	serial.runCommand(cmd, rsvMsg, sizeof(prtInfo));
}

/*
*		@brief		������ ���� ��ȣ �۽��� �޼���
*/
void DataCommunication::Command_Signal()
{
	char rsvMsg[5];
	PROTOCOL_INFO* ptr = &prtInfo;
	byte* cmd = (byte*)ptr;
	prtInfo.data.command = 0x03;		// 0x03 : ������ ���� ��ȣ ���Ž� ����

	serial.runCommand(cmd, rsvMsg, sizeof(prtInfo));
}
unsigned char DataCommunication::crcCalculate8_SAE_J1850(const unsigned char* ptr, int length)
{
	unsigned crc = 0xFF;

	while (length--)
	{
		unsigned char index = ((crc) ^ *ptr);
		crc = crc_table_[index];
		ptr++;
	}
	return ~crc;
}


int DataCommunication::serialConnect(char* _portNum)
{
	if (!serial.OpenPort(_portNum)) //��Ʈ�� �����ϰ� ���¿� �����Ͽ����� fail�� ��ȯ�Ѵ�.
		return RETURN_FAIL;

	serial.ConfigurePort(CBR_115200, 8, FALSE, NOPARITY, ONESTOPBIT); //��Ʈ �⺻���� �����Ѵ�.
	serial.SetCommunicationTimeouts(0, 0, 0, 0, 0); //Timeout�� ����

	return RETURN_SUCCESS;
}


void DataCommunication::sendPacket(PROTOCOL_INFO sendMsg)
{
}

void DataCommunication::disconnect()
{
	serial.ClosePort();
}




