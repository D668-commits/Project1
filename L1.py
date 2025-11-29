import numpy as np
import matplotlib.pyplot as plt

# Настройки для корректного отображения графиков
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.ioff()  # Режим отключения интерактивного режима для контроля отображения

class SignalProcessor:
    def __init__(self, N=50):
        self.N = N
        self.fs = 50  # Частота дискретизации
        self.t = np.linspace(0, 1, N, endpoint=False)
        self.x = np.linspace(0, 2*np.pi, N)
        
    def generate_signals(self, amplitudes=[1, 3], frequencies=[3, 1]):
        """Генерация синусоидальных сигналов"""
        self.signals = []
        for i in range(len(amplitudes)):
            signal_time = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * self.t)
            self.signals.append(signal_time)
        
        self.combined_signal = np.sum(self.signals, axis=0)
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        return self.combined_signal
    
    def generate_noise(self, noise_type='равномерный', noise_params=[0, 4]):
        """Генерация шума"""
        self.noise_type = noise_type
        self.noise_params = noise_params
        
        if noise_type == 'равномерный':
            self.noise = np.random.uniform(noise_params[0], noise_params[1], self.N)
        elif noise_type == 'нормальный':
            self.noise = np.random.normal(noise_params[0], noise_params[1], self.N)
        
        return self.noise
    
    def create_mixtures(self):
        """Создание аддитивной и мультипликативной смесей"""
        self.additive_mixture = self.combined_signal + self.noise
        self.multiplicative_mixture = self.combined_signal * self.noise
        return self.additive_mixture, self.multiplicative_mixture
    
    def compute_spectrum_corrected(self, signal, fs=None):
        """КОРРЕКТНОЕ вычисление спектра с использованием fftshift"""
        if fs is None:
            fs = self.fs
            
        n = len(signal)
        spectrum = np.fft.fft(signal) / n
        freq = np.fft.fftfreq(n, d=1/fs)
        
        # Применяем fftshift для корректного порядка частот
        spectrum_shifted = np.fft.fftshift(spectrum)
        freq_shifted = np.fft.fftshift(freq)
        
        return freq_shifted, np.abs(spectrum_shifted)
    
    def apply_filter(self, input_signal, center_freqs, bandwidth=1):
        """Применение прямоугольного фильтра"""
        n = len(input_signal)
        freq = np.fft.fftfreq(n, d=1/self.fs)
        
        # Создаем комбинированный фильтр
        combined_filter = np.zeros(n, dtype=complex)
        
        for center_freq in center_freqs:
            mask = (np.abs(freq - center_freq) <= bandwidth/2) | (np.abs(freq + center_freq) <= bandwidth/2)
            combined_filter[mask] = 1.0
        
        # Применяем фильтр в частотной области
        spectrum = np.fft.fft(input_signal)
        filtered_spectrum = spectrum * combined_filter
        filtered_signal = np.fft.ifft(filtered_spectrum).real
        
        return filtered_signal, combined_filter, filtered_spectrum
    
    def calculate_snr(self, original_signal, noise):
        """Расчет отношения сигнал-шум"""
        signal_power = np.mean(original_signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db

def lab_1_analysis():
    """Анализ для лабораторной работы 1"""
    print("=== ЛАБОРАТОРНАЯ РАБОТА 1 ===")
    print("Параметры: N=50, равномерный шум (0,4), 2 синусоиды: A=[1,3], f=[3,1] Гц")
    
    # Инициализация процессора сигналов
    sp = SignalProcessor(N=50)
    
    # Генерация сигналов и шума
    combined_signal = sp.generate_signals(amplitudes=[1, 3], frequencies=[3, 1])
    noise = sp.generate_noise(noise_type='равномерный', noise_params=[0, 4])
    additive_mix, multiplicative_mix = sp.create_mixtures()
    
    # === ГРАФИК 1: Временные реализации ===
    plt.figure(1, figsize=(15, 10))
    
    # График шума
    plt.subplot(2, 3, 1)
    plt.plot(sp.t, noise)
    plt.title('Равномерный шум (0, 4)')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # Графики отдельных синусоид
    plt.subplot(2, 3, 2)
    for i, sig in enumerate(sp.signals):
        plt.plot(sp.t, sig, label=f'Синусоида {i+1}\n(A={sp.amplitudes[i]}, f={sp.frequencies[i]} Гц)')
    plt.title('Отдельные синусоиды')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # График суммы синусоид
    plt.subplot(2, 3, 3)
    plt.plot(sp.t, combined_signal)
    plt.title('Сумма синусоид')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # График аддитивной смеси
    plt.subplot(2, 3, 4)
    plt.plot(sp.t, additive_mix)
    plt.title('Аддитивная смесь')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # График мультипликативной смеси
    plt.subplot(2, 3, 5)
    plt.plot(sp.t, multiplicative_mix)
    plt.title('Мультипликативная смесь')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # Спектральный анализ
    plt.subplot(2, 3, 6)
    freq_mix, spec_mix = sp.compute_spectrum_corrected(additive_mix)
    freq_sig, spec_sig = sp.compute_spectrum_corrected(combined_signal)
    freq_noise, spec_noise = sp.compute_spectrum_corrected(noise)
    
    plt.plot(freq_mix, spec_mix, label='Аддитивная смесь', linewidth=2)
    plt.plot(freq_sig, spec_sig, label='Сумма синусоид', linestyle='--')
    plt.plot(freq_noise, spec_noise, label='Шум', linestyle=':', alpha=0.7)
    plt.title('Спектры сигналов')
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.xlim(-20, 20)
    
    plt.tight_layout()
    plt.show()
    
    # Проверка обратного преобразования Фурье
    spectrum = np.fft.fft(additive_mix)
    reconstructed = np.fft.ifft(spectrum).real
    
    print("\nПроверка обратного преобразования Фурье:")
    print(f"Максимальная ошибка восстановления: {np.max(np.abs(additive_mix - reconstructed)):.2e}")
    
    # Расчет ОСШ
    snr_original = sp.calculate_snr(combined_signal, noise)
    print(f"Исходное ОСШ: {snr_original:.2f} дБ")
    
    # === ГРАФИК 2: Сравнение оригинального и восстановленного сигналов ===
    plt.figure(2, figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sp.t, additive_mix)
    plt.title('Оригинальная аддитивная смесь')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.subplot(1, 2, 2)
    plt.plot(sp.t, reconstructed)
    plt.title('Восстановленный сигнал (IFFT)')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.tight_layout()
    plt.show()
