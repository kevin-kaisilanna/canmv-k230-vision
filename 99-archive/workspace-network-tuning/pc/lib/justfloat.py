import struct

class JustFloatEncoder:
    """JustFloat协议编码器"""
    TAIL = b'\x00\x00\x80\x7f'  # 协议规定的帧尾
    
    @staticmethod
    def encode(channels):
        """
        编码浮点数组为JustFloat协议字节流
        :param channels: 浮点数列表
        :return: 编码后的字节流
        """
        # 将所有浮点数打包为小端字节序
        data = struct.pack('<' + 'f' * len(channels), *channels)
        return data + JustFloatEncoder.TAIL
    