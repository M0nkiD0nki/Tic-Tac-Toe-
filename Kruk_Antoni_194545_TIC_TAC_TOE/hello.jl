
# Sama gra w kołko i krzyzyk została zaimplementowana poprzez chat GPT. Wizualizacja oraz obsługiwanie myszy było inspirowane rozwiazaniami ze strony Makie pod adresem ,,https://docs.makie.org/v0.21/explanations/events" ,w funckji szukanie ruchu otrzymalem pomoc w sprawie watkow od chata oraz z filmu ,,https://www.youtube.com/watch?v=kX6_iY_BtG8&t=865s" ,o algorytmie nauczylem sie z filimku ,,https://www.youtube.com/watch?v=l-hh51ncgDI" 

using Printf
using Random
using Base.Threads
using GLMakie


const SIZE = 20                
const WIN_CONDITION = 5        
const RANGE_RADIUS = 2        
const MAX_DEPTH_START = 2      
const MAX_DEPTH_END = 4   

# Wizualizacja w oknie
grid_size = SIZE
player_positions = Observable(Vector{Point2f}())
ai_positions = Observable(Vector{Point2f}())

fig, ax, p = scatter(player_positions; marker = :xcross, markersize = 20, color = :red)
scatter!(ax, ai_positions; marker = :circle, markersize = 20, color = :blue)

# Rysowanie lini na planszy
for i in 0:grid_size
    # Pionowe linie 
    lines!(ax, [(i, 0), (i, grid_size)]; color = :gray, linewidth = 1, linestyle = :dash)
    # Poziome linie
    lines!(ax, [(0, i), (grid_size, i)]; color = :gray, linewidth = 1, linestyle = :dash)
end

# limity planszy
ax.limits = ((0, grid_size), (0, grid_size))



# Channel do komunikowania ruchow z modulu obslugi kliknięć do petli gry, roziwazanie zaproponowane przez chatGPT
move_channel = Channel{Tuple{Int, Int}}(1)

# Obsluga klikniec myszy
on(events(fig).mousebutton) do event
    if event.button == Mouse.left && event.action == Mouse.press
        # Pobieranie pozycji 
        mouse_pos = mouseposition(ax)
        
        # Zaokroglanie koordynatow 
        floored_x = floor(Int, mouse_pos[1])
        floored_y = floor(Int, mouse_pos[2])
        if  floored_x ≥ 0 && floored_y ≥ 0 && floored_x < grid_size && floored_y < grid_size
            # Zmiana indexoawania zamiast od 0 jest od 1
            col = floored_x + 1
            row = floored_y + 1  
            
            # wrzucanie do move_channel koordynatow
            put!(move_channel, (row, col))
            println("Ruch gracza: row=$row, col=$col")
            return Consume(true)
        end
    end
    return Consume(false)
end

display(fig)

#Implementacja gry w kolko i krzyzyk

# Tworzenie pustej planszy
function create_board(size)
    return fill(' ', size, size)
end

# Sprawdzanie, czy jest zwyciezca
function check_winner(board, player)
    # Sprawdza wiersze i kolumny
    for i in 1:SIZE
        for j in 1:(SIZE - WIN_CONDITION + 1)
            if all(board[i, j:j + WIN_CONDITION - 1] .== player) || 
               all(board[j:j + WIN_CONDITION - 1, i] .== player)
                return true
            end
        end
    end

    # Sprawdza przekątne
    for i in 1:(SIZE - WIN_CONDITION + 1)
        for j in 1:(SIZE - WIN_CONDITION + 1)
            if all([board[i + k, j + k] == player for k in 0:(WIN_CONDITION - 1)]) ||
               all([board[i + k, j + WIN_CONDITION - k - 1] == player for k in 0:(WIN_CONDITION - 1)])
                return true
            end
        end
    end
    return false
end

# Sprawdzanie remisu
function is_draw(board)
    return all(board .!= ' ')
end

# Ocenianie planszy i punktowanie ruchow
function board_score(board, player)
    score = 0
    for i in 1:SIZE
        for j in 1:(SIZE - WIN_CONDITION + 1)
            line = board[i, j:j + WIN_CONDITION - 1]
            score += checking_line(line, player)
            line = board[j:j + WIN_CONDITION - 1, i]
            score += checking_line(line, player)
        end
    end

    for i in 1:(SIZE - WIN_CONDITION + 1)
        for j in 1:(SIZE - WIN_CONDITION + 1)
            diag1 = [board[i + k, j + k] for k in 0:(WIN_CONDITION - 1)]
            diag2 = [board[i + k, j + WIN_CONDITION - k - 1] for k in 0:(WIN_CONDITION - 1)]
            score += checking_line(diag1, player)
            score += checking_line(diag2, player)
        end
    end
    return score
end

# Sprawdzanie zawartosci linii
function checking_line(line, player)
    opponent = (player == 'X') ? 'O' : 'X'
    player_count = count(c -> c == player, line)
    opponent_count = count(c -> c == opponent, line)
    
    #jezeli w linii jest tylko gracz zwraca wysokie punkty
    if player_count > 0 && opponent_count == 0
        return 3^player_count
    #jezeli w linii jest tylko przeciwnik zwraca niskie punkty
    elseif opponent_count > 0 && player_count == 0
        return -(3^opponent_count)
    else
        return 0
    end
end

# Algorytm Minimax z alpha-beta pruning oraz zaproponowane przez chat memoizacja w celu optymalizacji 
function minimax(board, depth, alpha, beta, maximizing, max_depth, moves, memo) 
    #memoizacja
    board_hash = hash(board)
    if haskey(memo, board_hash)
        return memo[board_hash]
    end

    #punktowanie konca gry 
    if check_winner(board, 'O')
        return 10_000 - depth
    elseif check_winner(board, 'X')
        return depth - 10_000
    elseif is_draw(board) || depth >= max_depth
        return board_score(board, 'O')
    end

    #Maksymalizowanie ruchu 'O' (przeciwnika)
    if maximizing
        max_eval = -Inf
        #Symulacja ruchu
        for (row, col) in nearest_center(possible_moves(board, moves, depth))
            board[row, col] = 'O'
            eval = minimax(board, depth + 1, alpha, beta, false, max_depth, moves, memo)
            #czyszczenie ruchu
            board[row, col] = ' '
            #aktualizacja ruchu
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha
                break
            end
        end
        memo[board_hash] = max_eval
        return max_eval
    else
        #Minimalizowanie gracza 'X'
        min_eval = Inf
        for (row, col) in nearest_center(possible_moves(board, moves, depth))
            board[row, col] = 'X'
            eval = minimax(board, depth + 1, alpha, beta, true, max_depth, moves, memo)
            board[row, col] = ' '
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha
                break
            end
        end
        memo[board_hash] = min_eval
        return min_eval
    end
end

# Wybieranie ruchow na podstawie bliskosci od srodka planszy
function nearest_center(moves)
    return sort(moves, by = x -> abs(x[1] - SIZE ÷ 2) + abs(x[2] - SIZE ÷ 2))
end

# Zmiana glebokosci szukania drzewa na podstawie etapu gry
function adjust_depth(moves_counter)
    if moves_counter > SIZE^2 * 0.9
        return MAX_DEPTH_END
    elseif moves_counter > SIZE^2 * 0.5
        return MAX_DEPTH_START + 1
    else
        return MAX_DEPTH_START
    end
end

# tworzenie obszarow na mozliwe ruchy wokol wykonanych ruchow 
function possible_moves(board, moves, moves_counter)
    radius = moves_counter > SIZE^2 * 0.6 ? RANGE_RADIUS + 1 : RANGE_RADIUS
    #przechowywanie mozliwych ruchow 
    move_set = Set()
    for (x, y) in moves
        for i in max(1, x - radius):min(SIZE, x + radius)
            for j in max(1, y - radius):min(SIZE, y + radius)
                if board[i, j] == ' '
                    push!(move_set, (i, j))
                end
            end
        end
    end
    return collect(move_set)
end

# Sprawdzanie czy dany ruch da wygrana 
function can_win(board, player, row, col)
    board[row, col] = player
    is_win = check_winner(board, player)
    board[row, col] = ' '  
    return is_win
end

# Znajdowanie najlepszego ruchu na podstawie minmax oraz funkcji pomocniczych 
function find_best_move(board, max_depth, moves, moves_counter)
    # Sprawdzanie czy jest mozliwa wygrana w jednym ruchu 
    for (row, col) in possible_moves(board, moves, moves_counter)
        if can_win(board, 'O', row, col)
            return (row, col)
        end
    end

    # Blokowanie jezeli przeciwnik jest bardzo blisko wygranej 
    for (row, col) in possible_moves(board, moves, moves_counter)
        if can_win(board, 'X', row, col)
            return (row, col)
        end
    end

    # Rownolegly min max
    best_move = Ref((-1, -1))
    #Przwchowywanie najlepszego wyniku
    best_value = Threads.Atomic{Float64}(-Inf)

    @threads for (row, col) in possible_moves(board, moves, moves_counter)
        #Kopia planszy aby nie wplywac na watki
        local_board = copy(board)
        #Symulacja ruchu
        local_board[row, col] = 'O'
        move_value = minimax(local_board, 0, -Inf, Inf, false, max_depth, moves, Dict())
        # println(Threads.threadid())
        if move_value > best_value[]
            atomic_max!(best_value, move_value)
            best_move[] = (row, col)
        end
    end

    return best_move[]
end

# glowna petla gry
function play_game()
    board = create_board(SIZE)
    current_player = 'X'
    moves_counter = 0
    #lista ruchow
    moves = Vector{Tuple{Int, Int}}()  

    while true
        #Obsluga gracza
        if current_player == 'X'
            println("Gracz $current_player, wykonaj ruch!")
            
            while true
                row, col = take!(move_channel)
                if board[row, col] != ' '
                    println("Nieprawidlowy ruch! Sprobuj ponownie!")
                    continue
                else
                    board[row, col] = current_player
                    x = col - 1 + 0.5
                    y = row - 1 + 0.5
                    push!(player_positions[], Point2f(x, y))
                    notify(player_positions)
                    push!(moves, (row, col))
                    moves_counter += 1
                    break
                end
            end
        else
            #Ruch przeciwnika
            println("Przeciwnik mysli")
            max_depth = adjust_depth(moves_counter)
            row, col = find_best_move(board, max_depth, moves, moves_counter)
            board[row, col] = current_player
            println("Przeciwnik wykonal ruch: ($row, $col)")
            x = col - 1 + 0.5
            y = row - 1 + 0.5
            push!(ai_positions[], Point2f(x, y))
            notify(ai_positions)
            push!(moves, (row, col))
            moves_counter += 1
        end

        if check_winner(board, current_player)
            println("WYGRYWA GRACZ $current_player !!")
            display(fig)
            wait()
            break
        elseif is_draw(board)
            println("REMIS!")
            display(fig)
            wait()
            break
        end

        current_player = (current_player == 'X') ? 'O' : 'X'
    end
end

# Start gry
play_game()
