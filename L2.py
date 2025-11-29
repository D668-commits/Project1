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
def lab_2_analysis():
    """Анализ для лабораторной работы 2"""
    print("\n" + "="*50)
    print("=== ЛАБОРАТОРНАЯ РАБОТА 2 ===")
    print("Параметры: N=50, нормальный шум (0,4), 2 синусоиды: A=[1,3], f=[3,1] Гц")
    
    # Инициализация процессора сигналов
    sp = SignalProcessor(N=50)
    
    # Генерация сигналов и шума
    combined_signal = sp.generate_signals(amplitudes=[1, 3], frequencies=[3, 1])
    noise = sp.generate_noise(noise_type='нормальный', noise_params=[0, 4])
    additive_mix, _ = sp.create_mixtures()
    
    # === ГРАФИК 3: Временные реализации ===
    plt.figure(3, figsize=(15, 12))
    
    # Временные реализации
    plt.subplot(3, 2, 1)
    plt.plot(sp.t, sp.signals[0])
    plt.title(f'Синусоида 1: {sp.amplitudes[0]}×sin(2π×{sp.frequencies[0]}×t)')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.subplot(3, 2, 2)
    plt.plot(sp.t, sp.signals[1])
    plt.title(f'Синусоида 2: {sp.amplitudes[1]}×sin(2π×{sp.frequencies[1]}×t)')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.subplot(3, 2, 3)
    plt.plot(sp.t, combined_signal)
    plt.title('Сумма синусоид')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.subplot(3, 2, 4)
    plt.plot(sp.t, noise)
    plt.title('БГШ (μ=0, σ=4)')  # ИСПРАВЛЕНА ОШИБКА - убрана f-строка
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.subplot(3, 2, 5)
    plt.plot(sp.t, additive_mix)
    plt.title('Аддитивная смесь сигнала и шума')
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    # Спектральный анализ
    freq_sig1, spec_sig1 = sp.compute_spectrum_corrected(sp.signals[0])
    freq_sig2, spec_sig2 = sp.compute_spectrum_corrected(sp.signals[1])
    freq_noise, spec_noise = sp.compute_spectrum_corrected(noise)
    
    plt.subplot(3, 2, 6)
    plt.plot(freq_sig1, spec_sig1, label=f'Синусоида {sp.frequencies[0]} Гц', linewidth=2)
    plt.plot(freq_sig2, spec_sig2, label=f'Синусоида {sp.frequencies[1]} Гц', linewidth=2)
    plt.plot(freq_noise, spec_noise, label='БГШ', alpha=0.7)
    plt.title('Спектры компонентов')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.xlim(-10, 10)
    
    plt.tight_layout()
    plt.show()
    
    # Фильтрация сигнала
    filtered_signal, combined_filter, filtered_spectrum = sp.apply_filter(
        additive_mix, center_freqs=sp.frequencies, bandwidth=1.0
    )
    
    # === ГРАФИК 4: Фильтрация сигнала ===
    plt.figure(4, figsize=(15, 10))
    
    # Спектр до фильтрации
    plt.subplot(2, 2, 1)
    freq_full = np.fft.fftfreq(len(additive_mix), d=1/sp.fs)
    spectrum_full = np.abs(np.fft.fft(additive_mix) / len(additive_mix))
    plt.plot(np.fft.fftshift(freq_full), np.fft.fftshift(spectrum_full))
    plt.title('Спектр аддитивной смеси')
    plt.grid(True)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.xlim(-10, 10)
    
    # АЧХ фильтра
    plt.subplot(2, 2, 2)
    plt.plot(np.fft.fftshift(freq_full), np.fft.fftshift(np.abs(combined_filter)))
    plt.title('АЧХ комбинированного фильтра')
    plt.grid(True)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Коэффициент передачи')
    plt.xlim(-10, 10)
    
    # Спектр после фильтрации
    plt.subplot(2, 2, 3)
    plt.plot(np.fft.fftshift(freq_full), np.fft.fftshift(np.abs(filtered_spectrum / len(additive_mix))))
    plt.title('Отфильтрованный спектр')
    plt.grid(True)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Амплитуда')
    plt.xlim(-10, 10)
    
    # Сравнение сигналов до и после фильтрации
    plt.subplot(2, 2, 4)
    plt.plot(sp.t, additive_mix, label='Исходная смесь', alpha=0.7)
    plt.plot(sp.t, filtered_signal, label='Отфильтрованный сигнал', linewidth=2)
    plt.plot(sp.t, combined_signal, label='Исходный сигнал', linestyle='--')
    plt.title('Сравнение сигналов')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    
    plt.tight_layout()
    plt.show()
    
    # Расчет ОСШ до и после фильтрации
    snr_before = sp.calculate_snr(combined_signal, noise)
    filtered_noise = filtered_signal - combined_signal
    snr_after = sp.calculate_snr(combined_signal, filtered_noise)
    
    print("\nРезультаты фильтрации:")
    print(f"Исходное ОСШ: {snr_before:.2f} дБ")
    print(f"ОСШ после фильтрации: {snr_after:.2f} дБ")
    print(f"Улучшение ОСШ: {snr_after - snr_before:.2f} дБ")

# Запуск анализа для обеих лабораторных работ
if __name__ == "__main__":
    print("Запуск объединенного проекта для лабораторных работ 1 и 2")
    print("Графики будут отображаться последовательно")
    print("Закрывайте каждое окно с графиком для перехода к следующему\n")
    
    lab_2_analysis()
