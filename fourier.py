import numpy as np # для работы с массивами и матрицами 
import matplotlib.pyplot as plt # для визуализации информации 
import sympy as sp # для аналитической математики 
import streamlit as st # библиотека для создания дашборда 
import time # для времени 

def fourier(): 
    try: 
        # для настройки страницы дашборда 
        st.set_page_config(page_title="Дашборд для ряда Фурье", layout="wide")
        st.title("Интерактивное разложение в ряд Фурье")
        st.markdown("Этот дашборд аналитически считает коэффициенты и строит приближение ряда Фурье для любой функции.")

        st.sidebar.header("Настройки функции")
        f_input = st.sidebar.text_input("Введите функцию f(x):", value="x") # как пример функция от x 


        st.sidebar.subheader("Интервал")
        interval_type = st.sidebar.selectbox("Тип интервала:", ["[-π, π]", "Выбрать свой"])
        # исходя из того, что выбрал пользователь 
        if interval_type == "[-π, π]": # выбрал от минус пи до пи 
            x_min, x_max = -np.pi, np.pi # для генерации точек графика 
            x_min_sym, x_max_sym = -sp.pi, sp.pi # для аналитических вычислений 
        else:
            x_min = st.sidebar.number_input("Нижняя граница (x_min):", value=-1.0) # нижняя граница пользовательского интервала 
            x_max = st.sidebar.number_input("Верхняя граница (x_max):", value=1.0)
            x_min_sym, x_max_sym = sp.nsimplify(x_min), sp.nsimplify(x_max) # защита от дробных ошибок 

        st.sidebar.subheader("Число гармоник N")
        N_max = st.sidebar.slider("Максимальное число гармоник N:", min_value=1, max_value=50, value=10)




        


        # аналитические вычисление ряда Фурье 
        x = sp.symbols('x') # Объявляем x как символ 
        n = sp.symbols('n', integer=True, positive=True) # объявляем n как целое и положительное 
        
        parsed_function = sp.sympify(f_input) # Достаем значение функции в выражение sympy

        if callable(parsed_function) or isinstance(parsed_function, type): 
            raise TypeError("Вы ввели имя функции без аргумента")
        else:
            function = parsed_function
        # найдем полупериод T = 2L или L = T/2
        T_value = float((x_max - x_min)) # для точек numpy 
        T_sympy = (x_max_sym - x_min_sym) # для sympy (интегрирования)


        a0 = (2 / T_sympy) * sp.integrate(function, (x, x_min_sym, x_max_sym))
        an = (2 / T_sympy) * sp.integrate(function * sp.cos(2 * sp.pi * n * x / T_sympy), (x, x_min_sym, x_max_sym))
        bn = (2 / T_sympy) * sp.integrate(function * sp.sin(2 * sp.pi * n * x / T_sympy), (x, x_min_sym, x_max_sym))


        # превращаем абстракные математические формулы из библиотеки sympy в числовые функции для numpy 
        # необходимо для того, чтобы быстро отображать графики так как sympy работает с формулами как с текстом 
        f_num = sp.lambdify(x, function, 'numpy')
        an_num = sp.lambdify(n, an, 'numpy')
        bn_num = sp.lambdify(n, bn, 'numpy')
        a0_value = float(a0)


        col1, col2 = st.columns([1, 2]) # вывод аналитики на дашборд 

        with st.expander("Математическое описание для ряда Фурье"):
            st.markdown("""
                ### Описание ряда Фурье:
                **Ряд Фурье** - это математический способ представления сложной периодической функции 
                в виде суммы более простых гармонических колебаний - sin и cos.
            """)

            st.latex(r"f(x) \approx \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos\left(\frac{2n\pi x}{T}\right) + b_n \sin\left(\frac{2n\pi x}{T}\right) \right)")
            

            st.markdown("""
                ### Коэффициенты (громкость гармоник):
                * **$a_0$** — постоянная составляющая:
                """)
            st.latex(r"a_0 = \frac{2}{T} \int_{x_{min}}^{x_{max}} f(x) \, dx")
    
            st.markdown("""
                * **$a_n$** — амплитуды косинусоид:
                """)
            st.latex(r"a_n = \frac{2}{T} \int_{x_{min}}^{x_{max}} f(x) \cos\left(\frac{2n\pi x}{T}\right) \, dx")
                
            st.markdown("""
                * **$b_n$** — амплитуды синусоид:
                """)
            st.latex(r"b_n = \frac{2}{T} \int_{x_{min}}^{x_{max}} f(x) \sin\left(\frac{2n\pi x}{T}\right) \, dx")
        with col1:
            st.subheader("Аналитические коэффициенты")
            st.latex(f"a_0 = {sp.latex(sp.simplify(a0))}") # делаем красивый вид функции
            st.latex(f"a_n = {sp.latex(sp.simplify(an))}")
            st.latex(f"b_n = {sp.latex(sp.simplify(bn))}")

            if bn == 0: 
                st.success("Функция является четной, только косинусу")
            elif an == 0 and a0 == 0: 
                st.success("Функция является нечетной, только синусы")
            else:
                st.info("Функция общего вида (и синусы, и косинусы)")

        with col2:    
            st.subheader("График аппроксимации")   
            animation_fourier = st.button("Анимация") # кнопка для анимации 
            x_vals = np.linspace(x_min, x_max, 1000) # создаем сетку точек от - пи до пи, 1000 точек

            plot_container = st.empty() # контейнер для обновления графика 

            def draw(max_n_current):
                fig, ax = plt.subplots(figsize=(10, 5)) # окно для графика 

                y_orig = f_num(x_vals) # вычисляем координаты по y
                if isinstance(y_orig, (int, float)): # защита от константы 
                    y_orig = np.full_like(x_vals, y_orig)

                # строим исходную функцию на графике 
                ax.plot(x_vals, y_orig, label=f'Исходная функция f(x) = {f_input}', color='black', lw=2.5, linestyle='--')

                # функция апроксимации будет начинаться со стандартного коэффициента 
                function_approximation = np.full_like(x_vals, a0_value / 2, dtype=np.float64)
                
                for num_n in range(1, max_n_current +1): # используем стандартный ряд Фурье от 1 до бесконечности
                    try: # проверяем значения коэффициентов, чтобы не было ошибки
                        an_value = float(an_num(num_n))
                    except Exception as e: 
                        an_value = 0.0 

                    try: 
                        bn_value = float(bn_num(num_n))
                    except Exception as e: 
                        bn_value = 0.0 


                    # считаем функцию апроксимации 
                    function_approximation += an_value * np.cos(2 * np.pi * num_n * x_vals / T_value) + bn_value * np.sin(2 * np.pi * num_n * x_vals / T_value)
                        # приближение ряда Фурье 
                ax.plot(x_vals, function_approximation.copy(), label=f'Ряд Фурье (N={max_n_current})', alpha=0.8) 

            
                ax.set_xlabel('x')
                ax.set_ylabel('f(x)')
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend(loc='best')
                return fig 
            
            if animation_fourier:
                step = 1 if N_max <= 15 else (2 if N_max <= 30 else 3) # если гармоник много, сокращаем время

                for frame in range(1, N_max + 1, step): 
                    fig = draw(frame)
                    plot_container.pyplot(fig) # перерисовываем график 
                    plt.close(fig) # Очистили память от предыдущего графика 
                    time.sleep(0.1) 


                # финальный кадр для графика 
                fig = draw(N_max)
                plot_container.pyplot(fig)
                plt.close(fig)
            else: 
                # отрисовываем сразу, если кнопка не нажата 
                fig = draw(N_max)
                plot_container.pyplot(fig)
                plt.close(fig)
    except Exception as e: 
        st.error(f"Ошибка математического парсинга или интеграции. Убедитесь, что функция введена корректно. Ошибка: {e}")
        st.info("Подсказка: используйте 'x' в качестве переменной, '**' для возведения в степень (например, `x**2`), `sp.sin(x)` или `sp.cos(x)` для тригонометрии.")

if __name__ == "__main__": 
    fourier()


