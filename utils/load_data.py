import pickle
import numpy as np
import os
import glob



def load_articles(obj):
    print('Dataset: ', obj)
    print("loading news articles")

    train_dict = pickle.load(open('dataset/news_articles/' + obj + '_train_re.pkl', 'rb'))
    test_dict = pickle.load(open('dataset/news_articles/' + obj + '_test_re.pkl', 'rb'))

    restyle_dict = pickle.load(open('dataset/emotions/' + obj+ '_test_anger.pkl', 'rb'))
    # alternatively, switch to loading other adversarial test sets with '_test_adv_[B/C/D].pkl'

    x_train, y_train, z_train = train_dict['news'], train_dict['labels'], train_dict["explanation"] 
    x_test, y_test, z_test = test_dict['news'], test_dict['labels'], test_dict["explanation"]

    x_test_res = restyle_dict['news']

    return x_train, x_test, x_test_res, y_train, y_test, z_train, z_test


def load_reframing(obj):
    print("loading news augmentations")
    print('Dataset: ', obj)

    restyle_dict_train1_1 = pickle.load(open('dataset/reframings/' + obj+ '_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open('dataset/reframings/' + obj+ '_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open('dataset/reframings/' + obj+ '_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open('dataset/reframings/' + obj+ '_train_sensational.pkl', 'rb'))

    finegrain_dict1 = pickle.load(open('dataset/veracity_attributions/' + obj+ '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open('dataset/veracity_attributions/' + obj+ '_fake_standards_neutral_sensational.pkl', 'rb'))

    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]


    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t


def load_emotion_tests(obj, emotions=None):
    """Load multiple emotion-specific test sets.

    Args:
        obj: dataset name (e.g., 'politifact')
        emotions: list of emotion names (e.g., ['anger', 'happiness']). If None or empty,
                  auto-detect all available emotions under dataset/emotions/.

    Returns:
        dict mapping emotion -> np.array of rewritten news texts
    """
    emotion_dir_candidates = [
        os.path.join('dataset', 'emotions'),
        os.path.join('data', 'emotion'),
    ]
    emotion_dir = next((d for d in emotion_dir_candidates if os.path.isdir(d)), emotion_dir_candidates[0])

    detected = []
    if not emotions:
        pattern = os.path.join(emotion_dir, f'{obj}_test_*.pkl')
        for path in glob.glob(pattern):
            base = os.path.basename(path)
            # filename pattern: {obj}_test_{emotion}.pkl
            try:
                emo = base.split('_test_')[1].rsplit('.pkl', 1)[0]
                if emo:
                    detected.append(emo)
            except Exception:
                continue
        emotions = sorted(list(set(detected)))
        print(f"Auto-detected emotions for testing: {emotions}")

    tests = {}
    for emo in emotions:
        path = os.path.join(emotion_dir, f'{obj}_test_{emo}.pkl')
        try:
            restyle_dict = pickle.load(open(path, 'rb'))
            tests[emo] = np.array(restyle_dict['news'])
        except FileNotFoundError:
            print(f"[WARN] Emotion test file not found: {path}")
        except Exception as e:
            print(f"[WARN] Failed to load emotion '{emo}': {e}")
    return tests
